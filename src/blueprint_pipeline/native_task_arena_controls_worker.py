"""Run the required task-neutral controls on one qualified native Arena cell.

The worker consumes a sealed scene packet plus two digest-bound runtime inputs:
the successful construction receipt and the control plan compiled from it.  It
does not name a scene, object class, joint, or policy.  Both controls execute
through the same native eight-dimensional action seam and deterministic scorer
that later policy episodes use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
RESULT_FILENAME = "native_task_arena_control_result.v1.json"
POSITION_ONLY_PREALIGN_ORIENTATION_TOLERANCE_RAD = 0.08
# C25 proved that pose reachability alone is not enough for contact.  The
# nominal jaw branch reached the 5 mm off-sim pose gate with panda_joint5 only
# 0.00053 rad from its limit, then the live PhysX controller saturated 14.5 mm
# from the handle.  The equivalent jaw branch carried a continuous
# prealign->approach->contact path when approach admitted the 0.04 rad-margin
# branch and contact required at least 0.005 rad of room.  These remain branch
# selection constraints only; native measured arrival and contact still gate
# the episode.
CONTROLS_APPROACH_PREFERRED_JOINT_MARGIN_RAD = 0.04
CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD = 0.005

# C28 sealed the decisive contact-entry evidence: three measured-miss-biased
# Cartesian attempts diverged 15.4 -> 28.5 -> 38.6 mm while the wrist swung
# up to 0.264 rad -- the live servo approaches the contact pose through a
# degenerate direction-space where no Cartesian correction can land, even
# though the multistart preflight holds an exact interior-margin solution for
# that pose on the branch it selected for approach->contact continuity.  The
# entry into contact_open is therefore replayed as a bounded joint micro-path
# between the two solved same-branch postures.  The contact_open arrival gate,
# contact scoring, and collision predicates are unchanged: if the solved
# branch does not put the measured fingertip at the handle, the phase still
# fails honestly.
CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID = "contact_open_branch_replay"
CONTACT_ENTRY_BRANCH_REPLAY_MAX_STEP_RAD = 0.05
CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS = 5
# Raised once the row size became actuator-bound: a 0.6 rad reorientation
# at a wrist-feasible 0.005 rad per step simply needs ~124 rows.  The task
# step budget remains the real limiter; this only stops a runaway.
CONTACT_ENTRY_BRANCH_REPLAY_MAX_TOTAL_ROWS = 240


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena_controls:{phase}:{status}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _persist(output: Path, result: dict[str, Any]) -> None:
    # Normalise before digesting. This runs from a `finally`, and
    # `_canonical_digest` refuses values json cannot encode -- a stray warp
    # array or Path would raise *inside* the handler, replace the real
    # exception and leave a paid run with no receipt at all. Passing
    # `default=str` to the write alone is not enough, because the digest is
    # computed first. Normalising both also makes the digest describe exactly
    # the bytes on disk.
    normalised = json.loads(json.dumps(result, default=str))
    normalised["result_digest"] = _canonical_digest(
        normalised, field="result_digest"
    )
    result["result_digest"] = normalised["result_digest"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalised, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_manifest(runtime: Path) -> dict[str, Any]:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    manifest = json.loads(
        (runtime / "adp_arena_provider_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or manifest.get("execution_mode") != "controls"
        or manifest.get("policy_candidate_id") is not None
        or manifest.get("candidate_policy_queried") is not False
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("native_task_controls_manifest_invalid")
    return manifest


def _verified_runtime_inputs(
    runtime: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    rows = manifest.get("bound_runtime_inputs")
    if not isinstance(rows, list):
        raise RuntimeError("native_task_controls_runtime_inputs_invalid")
    verified: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("native_task_controls_runtime_inputs_invalid")
        relative = str(row.get("relative_path") or "")
        path = runtime / relative
        if (
            not relative.startswith("runtime_inputs/")
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise RuntimeError(
                f"native_task_controls_runtime_input_identity_mismatch:{relative}"
            )
        verified[Path(relative).name] = path
    required = {
        "native_task_arena_construction_result.v1.json",
        "adp_task_control_plan.v1.json",
    }
    if set(verified) != required:
        raise RuntimeError("native_task_controls_runtime_inputs_incomplete")
    return verified


def _bound_digest(value: Any) -> bool:
    """A digest relation only holds when both sides actually carry a digest."""
    return isinstance(value, str) and value.startswith("sha256:") and len(value) > 7


def _construction_global_ik_joint_targets(
    *, construction: Mapping[str, Any], control_plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Bind the joint branches construction actually selected to controls.

    PINK is a local differential IK solver.  The construction worker therefore
    runs a bounded multi-start preflight and replays a selected joint target
    when one exists.  Controls must use the same branch; starting local IK over
    from the position-only prealignment can converge against different joint
    limits even though both workers consume the same Cartesian phase plan.
    """

    preflight = construction.get("pink_global_ik_preflight")
    if preflight is None:
        return []
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("schema_version")
        != "native_task_pink_global_ik_preflight.v1"
        or not isinstance(preflight.get("phases"), list)
    ):
        raise RuntimeError("native_task_controls_global_ik_preflight_invalid")
    actions = control_plan.get("scripted_positive_actions")
    phase_results = construction.get("phase_results")
    if not isinstance(actions, list) or not isinstance(phase_results, list):
        raise RuntimeError("native_task_controls_global_ik_control_plan_invalid")
    actions_by_phase: dict[str, Mapping[str, Any]] = {}
    for raw in actions:
        if not isinstance(raw, Mapping) or raw.get("mode") != "ik_pose":
            continue
        phase_id = str(raw.get("phase_id") or "")
        if not phase_id or phase_id in actions_by_phase:
            raise RuntimeError("native_task_controls_global_ik_control_plan_invalid")
        actions_by_phase[phase_id] = raw
    results_by_phase: dict[str, Mapping[str, Any]] = {}
    for raw in phase_results:
        if not isinstance(raw, Mapping):
            raise RuntimeError("native_task_controls_global_ik_preflight_invalid")
        phase_id = str(raw.get("phase_id") or "")
        if not phase_id or phase_id in results_by_phase:
            raise RuntimeError("native_task_controls_global_ik_preflight_invalid")
        results_by_phase[phase_id] = raw

    rows: list[dict[str, Any]] = []
    seen_phases: set[str] = set()
    for phase in preflight["phases"]:
        if not isinstance(phase, Mapping):
            raise RuntimeError("native_task_controls_global_ik_preflight_invalid")
        selected = phase.get("selected")
        if selected is None:
            continue
        phase_id = str(phase.get("phase_id") or "")
        action = actions_by_phase.get(phase_id)
        # Controls deliberately rename and offset its contact trajectory.  A
        # construction seed is reusable only when the control carries the
        # same phase identity and exact measured target pose.  Unmatched
        # construction phases remain local-IK controls; they are not errors.
        if action is None:
            continue
        phase_result = results_by_phase.get(phase_id)
        if (
            not phase_id
            or phase_id in seen_phases
            or not isinstance(selected, Mapping)
            or selected.get("solved") is not True
            or not isinstance(phase_result, Mapping)
        ):
            raise RuntimeError("native_task_controls_global_ik_binding_invalid")
        try:
            position = [
                float(value) for value in action["target_position_world_m"]
            ]
            quaternion = [
                float(value) for value in action["target_quaternion_world_xyzw"]
            ]
            joints = [
                float(value) for value in selected["joint_positions_rad"]
            ]
            construction_position = [
                float(value) for value in phase_result["target_position_world_m"]
            ]
            construction_quaternion = [
                float(value)
                for value in phase_result["target_orientation_world_xyzw"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_global_ik_binding_invalid"
            ) from exc
        if (
            len(position) != 3
            or len(joints) != 7
            or len(quaternion) != 4
            or position != construction_position
            or quaternion != construction_quaternion
            or not all(
                math.isfinite(value)
                for value in [*position, *joints, *quaternion]
            )
        ):
            raise RuntimeError("native_task_controls_global_ik_binding_invalid")
        rows.append(
            {
                "phase_id": phase_id,
                "target_position_world_m": position,
                "target_quaternion_world_xyzw": quaternion,
                "joint_positions_rad": joints,
            }
        )
        seen_phases.add(phase_id)
    return rows


def _control_plan_global_ik_joint_targets(
    *,
    servo: Any,
    control_plan: Mapping[str, Any],
    bound_targets: Sequence[Mapping[str, Any]],
    reference_seeds: Sequence[Sequence[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve exact controls poses that have no construction-selected branch."""

    actions = control_plan.get("scripted_positive_actions")
    if (
        not isinstance(actions, list)
        or not callable(getattr(servo, "solve_grasp_target_multistart", None))
        or not callable(getattr(servo, "read_arm_joint_positions", None))
    ):
        raise RuntimeError("native_task_controls_multistart_input_invalid")
    targets = [dict(row) for row in bound_targets]
    by_pose: dict[tuple[tuple[float, ...], tuple[float, ...]], dict[str, Any]] = {}
    for row in targets:
        try:
            key = (
                tuple(float(value) for value in row["target_position_world_m"]),
                tuple(
                    float(value)
                    for value in row["target_quaternion_world_xyzw"]
                ),
            )
            joints = [float(value) for value in row["joint_positions_rad"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_multistart_input_invalid"
            ) from exc
        if len(key[0]) != 3 or len(key[1]) != 4 or len(joints) != 7 or key in by_pose:
            raise RuntimeError("native_task_controls_multistart_input_invalid")
        by_pose[key] = row

    reference = [float(value) for value in servo.read_arm_joint_positions()]
    phases: list[dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, Mapping) or raw.get("mode") != "ik_pose":
            continue
        phase_id = str(raw.get("phase_id") or "")
        position_only_arrival = raw.get("position_only_arrival") is True
        try:
            position = [float(value) for value in raw["target_position_world_m"]]
            quaternion = [
                float(value) for value in raw["target_quaternion_world_xyzw"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_multistart_input_invalid"
            ) from exc
        key = (tuple(position), tuple(quaternion))
        existing = by_pose.get(key)
        if existing is not None:
            reference = [float(value) for value in existing["joint_positions_rad"]]
            phases.append(
                {
                    "phase_id": phase_id,
                    "status": "reused_bound_pose_solution",
                    "source_phase_id": existing["phase_id"],
                }
            )
            continue
        try:
            position_tolerance_m = float(raw["arrival_tolerance_m"])
            orientation_tolerance_rad = (
                POSITION_ONLY_PREALIGN_ORIENTATION_TOLERANCE_RAD
                if position_only_arrival
                else float(raw["arrival_orientation_tolerance_rad"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_multistart_tolerance_invalid"
            ) from exc
        if (
            not math.isfinite(position_tolerance_m)
            or position_tolerance_m <= 0.0
            or not math.isfinite(orientation_tolerance_rad)
            or orientation_tolerance_rad <= 0.0
        ):
            raise RuntimeError(
                "native_task_controls_multistart_tolerance_invalid"
            )
        solved = servo.solve_grasp_target_multistart(
            target_position_world_m=position,
            target_grasp_frame_quaternion_world_xyzw=quaternion,
            preferred_seeds=[reference, *reference_seeds],
            reference_joint_positions_rad=reference,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            preferred_minimum_joint_limit_margin_rad=(
                CONTROLS_APPROACH_PREFERRED_JOINT_MARGIN_RAD
                if phase_id == "approach"
                else 0.05
            ),
            required_minimum_joint_limit_margin_rad=(
                CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD
                if phase_id in {"contact_open", "contact_close"}
                else 0.0
            ),
        )
        if not isinstance(solved, Mapping):
            raise RuntimeError("native_task_controls_multistart_result_invalid")
        selected = solved.get("selected")
        phases.append(
            {
                "phase_id": phase_id,
                "position_only_arrival_gate": position_only_arrival,
                "full_pose_prepositioning_tolerance_rad": (
                    orientation_tolerance_rad
                    if position_only_arrival
                    else None
                ),
                **dict(solved),
            }
        )
        if not isinstance(selected, Mapping):
            continue
        joints = [float(value) for value in selected["joint_positions_rad"]]
        row = {
            "phase_id": phase_id,
            "target_position_world_m": position,
            "target_quaternion_world_xyzw": quaternion,
            "joint_positions_rad": joints,
        }
        targets.append(row)
        by_pose[key] = row
        reference = joints
    receipt = {
        "schema_version": "native_task_controls_global_ik_preflight.v1",
        "status": (
            "all_unique_poses_solved_or_bound"
            if all(
                phase.get("status") == "reused_bound_pose_solution"
                or phase.get("selected") is not None
                for phase in phases
            )
            else "partial"
        ),
        "phase_count": len(phases),
        "joint_target_count": len(targets),
        "phases": phases,
        "provider_mutation_performed": False,
        "physics_steps_performed": 0,
        "claim_boundary": (
            "off_sim_multistart_pose_ik_only;native_controls_remain_the_"
            "arrival_contact_dynamics_and_task_outcome_authority"
        ),
    }
    return targets, receipt


def _parallel_jaw_equivalent_quaternion_xyzw(
    quaternion_xyzw: Sequence[float],
) -> list[float]:
    """Swap interchangeable fingers without changing the approach direction.

    A parallel jaw is physically a line, not an arrow. Post-multiplying the
    commanded grasp orientation by a half turn about its local +Z approach
    axis preserves that approach axis and reverses local +X/+Y. The two
    commands therefore ask for the same fingertip centre and jaw line while
    exposing the other Franka wrist branch to the joint-limited solver.
    """

    try:
        x, y, z, w = [float(value) for value in quaternion_xyzw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "native_task_controls_parallel_jaw_quaternion_invalid"
        ) from exc
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        raise RuntimeError("native_task_controls_parallel_jaw_quaternion_invalid")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise RuntimeError("native_task_controls_parallel_jaw_quaternion_invalid")
    # Hamilton product q * [0, 0, 1, 0], with quaternions encoded xyzw.
    rotated = [y / norm, -x / norm, w / norm, -z / norm]
    # q and -q encode the same orientation. Canonicalise the sign so the
    # runtime-derived plan has one deterministic digest.
    for value in rotated:
        if abs(value) <= 1.0e-15:
            continue
        if value < 0.0:
            rotated = [-component for component in rotated]
        break
    return [0.0 if abs(value) <= 1.0e-15 else value for value in rotated]


def _quaternion_axis_world_xyzw(
    quaternion_xyzw: Sequence[float], axis_body: Sequence[float]
) -> list[float]:
    """Rotate one body-frame direction into world for an evidence receipt."""

    x, y, z, w = [float(value) for value in quaternion_xyzw]
    ax, ay, az = [float(value) for value in axis_body]
    # R(q) @ axis, expanded to avoid importing a simulator/math package in
    # the CPU-only bundle tests.
    return [
        (1.0 - 2.0 * (y * y + z * z)) * ax
        + 2.0 * (x * y - z * w) * ay
        + 2.0 * (x * z + y * w) * az,
        2.0 * (x * y + z * w) * ax
        + (1.0 - 2.0 * (x * x + z * z)) * ay
        + 2.0 * (y * z - x * w) * az,
        2.0 * (x * z - y * w) * ax
        + 2.0 * (y * z + x * w) * ay
        + (1.0 - 2.0 * (x * x + y * y)) * az,
    ]


def _parallel_jaw_equivalent_control_plan(
    control_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive and digest the finger-swapped equivalent of one sealed plan."""

    try:
        derived = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("native_task_controls_parallel_jaw_plan_invalid") from exc
    actions = derived.get("scripted_positive_actions")
    if not isinstance(actions, list) or not actions:
        raise RuntimeError("native_task_controls_parallel_jaw_plan_invalid")
    transformed = 0
    example: dict[str, Any] | None = None
    for action in actions:
        if not isinstance(action, dict) or action.get("mode") != "ik_pose":
            continue
        original = action.get("target_quaternion_world_xyzw")
        if original is None:
            continue
        equivalent = _parallel_jaw_equivalent_quaternion_xyzw(original)
        action["target_quaternion_world_xyzw"] = equivalent
        transformed += 1
        if example is None:
            original_axis = _quaternion_axis_world_xyzw(original, [0.0, 1.0, 0.0])
            equivalent_axis = _quaternion_axis_world_xyzw(
                equivalent, [0.0, 1.0, 0.0]
            )
            original_approach = _quaternion_axis_world_xyzw(
                original, [0.0, 0.0, 1.0]
            )
            equivalent_approach = _quaternion_axis_world_xyzw(
                equivalent, [0.0, 0.0, 1.0]
            )
            example = {
                "original_quaternion_world_xyzw": list(original),
                "equivalent_quaternion_world_xyzw": equivalent,
                "approach_axis_dot": sum(
                    left * right
                    for left, right in zip(
                        original_approach, equivalent_approach, strict=True
                    )
                ),
                "jaw_axis_dot": sum(
                    left * right
                    for left, right in zip(
                        original_axis, equivalent_axis, strict=True
                    )
                ),
            }
    if transformed < 1 or example is None:
        raise RuntimeError("native_task_controls_parallel_jaw_plan_invalid")
    source_digest = str(control_plan.get("plan_digest") or "")
    derived["runtime_control_variant"] = (
        "parallel_jaw_half_turn_about_local_approach_axis"
    )
    derived["runtime_control_variant_source_plan_digest"] = source_digest
    derived["runtime_control_variant_equivalence"] = {
        "schema_version": "native_task_parallel_jaw_equivalence.v1",
        "transformed_pose_count": transformed,
        "finger_identity_interchangeable": True,
        "jaw_axis_sign_is_a_label_not_a_measurement": True,
        "position_targets_unchanged": True,
        "local_half_turn_axis": "+z_approach",
        **example,
    }
    derived["plan_digest"] = _canonical_digest(derived, field="plan_digest")
    return derived


def _normalized_control_plan_for_execution(
    *, control_plan: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind validator-normalized executable bytes to the sealed input plan."""

    from blueprint_pipeline.adp009d_control_episode import (
        validate_task_control_plan,
    )

    normalized = validate_task_control_plan(control_plan, task_spec=task_spec)
    source_digest = str(control_plan.get("plan_digest") or "")
    normalized["runtime_normalized_source_plan_digest"] = source_digest
    normalized["runtime_normalization"] = {
        "schema_version": "native_task_control_plan_normalization.v1",
        "source_plan_digest": source_digest,
        "normalizer": (
            "blueprint_pipeline.adp009d_control_episode."
            "validate_task_control_plan"
        ),
        "claim_boundary": (
            "type_and_field_normalization_only;does_not_assert_pose_arrival_"
            "contact_or_task_success"
        ),
    }
    normalized["plan_digest"] = _canonical_digest(normalized, field="plan_digest")
    return normalized


def _with_contact_entry_branch_replay(
    *,
    control_plan: Mapping[str, Any],
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]],
    task_spec: Mapping[str, Any],
    actuator_feasible_step_rad: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enter contact_open on the solved branch instead of servoing onto it.

    Inserts a bounded joint micro-path (at most 0.05 rad per joint per row,
    plus settle copies) between the solved approach posture and the solved
    contact posture, immediately before the unchanged contact_open row.  The
    two postures come from the same multistart preflight that selected the
    branch for approach->contact continuity and interior joint-limit margin,
    so the interpolation stays on that branch.  Fails open to the previous
    behavior -- with a typed receipt -- when either posture is unsolved or
    the plan's step budget cannot absorb the rows.
    """

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    receipt: dict[str, Any] = {
        "schema_version": "native_task_controls_contact_entry_branch_replay.v1",
        "status": "not_applied",
        "reason": None,
        "source_control_plan_digest": plan.get("plan_digest"),
        "claim_boundary": (
            "bounded_same_branch_joint_micro_path_before_unchanged_contact_"
            "arrival_gate;native_controls_remain_the_arrival_contact_"
            "dynamics_and_task_outcome_authority"
        ),
    }
    actions = plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        receipt["reason"] = "scripted_positive_actions_invalid"
        return plan, receipt

    def _pose_key(row: Mapping[str, Any]) -> tuple | None:
        try:
            return (
                tuple(float(value) for value in row["target_position_world_m"]),
                tuple(
                    float(value)
                    for value in row["target_quaternion_world_xyzw"]
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    joints_by_pose: dict[tuple, list[float]] = {}
    for row in scripted_pose_joint_targets:
        key = _pose_key(row)
        try:
            joints = [float(value) for value in row["joint_positions_rad"]]
        except (KeyError, TypeError, ValueError):
            continue
        if key is not None and len(joints) == 7:
            joints_by_pose[key] = joints

    contact_index: int | None = None
    approach_key = None
    contact_key = None
    for index, raw in enumerate(actions):
        if not isinstance(raw, Mapping) or raw.get("mode") != "ik_pose":
            continue
        phase_id = str(raw.get("phase_id") or "")
        if phase_id == "approach":
            approach_key = _pose_key(raw)
        elif phase_id == "contact_open":
            contact_index = index
            contact_key = _pose_key(raw)
    if contact_index is None or approach_key is None or contact_key is None:
        receipt["reason"] = "approach_or_contact_phase_missing"
        return plan, receipt
    start = joints_by_pose.get(approach_key)
    end = joints_by_pose.get(contact_key)
    if start is None or end is None:
        receipt["reason"] = "approach_or_contact_solution_unsolved"
        return plan, receipt

    # These rows are commanded joint targets, so they bypass the servo's slew
    # and lead bounding entirely -- which is why C30's replay drove straight
    # into saturation by its eleventh row.  Size each row by what the slowest
    # participating actuator can actually track in one control step.
    per_joint_step = (
        [abs(float(value)) for value in actuator_feasible_step_rad]
        if actuator_feasible_step_rad
        else None
    )
    feasible_step = CONTACT_ENTRY_BRANCH_REPLAY_MAX_STEP_RAD
    if per_joint_step and len(per_joint_step) == len(start):
        moving = [
            step
            for step, s, e in zip(per_joint_step, start, end)
            if step > 0.0 and abs(e - s) > 1.0e-9
        ]
        if moving:
            feasible_step = min(feasible_step, min(moving))

    max_delta = max(abs(e - s) for s, e in zip(start, end))
    settle_rows = CONTACT_ENTRY_BRANCH_REPLAY_SETTLE_ROWS
    maximum_action_steps = task_spec.get("maximum_action_steps")
    settle = int(task_spec.get("settle_window_samples") or 0)
    planned_steps = 0
    contact_row = actions[contact_index]
    for raw in actions:
        if isinstance(raw, Mapping) and raw.get("mode") == "ik_pose":
            try:
                planned_steps += int(raw["maximum_steps"])
            except (KeyError, TypeError, ValueError):
                receipt["reason"] = "plan_step_budget_unreadable"
                return plan, receipt
        else:
            planned_steps += 1

    # A replayed entry lands the pose, so the Cartesian phase behind it only
    # has to confirm arrival rather than search for it.  Reclaiming that
    # budget is what buys the rows a feasible step size needs.
    reclaimed = 0
    try:
        contact_maximum = int(contact_row["maximum_steps"])
        contact_minimum = int(contact_row["minimum_steps"])
        contact_stability = int(contact_row["arrival_stability_steps"])
    except (KeyError, TypeError, ValueError):
        receipt["reason"] = "plan_step_budget_unreadable"
        return plan, receipt
    confirm_steps = max(contact_minimum, contact_stability) + 2
    if contact_maximum > confirm_steps:
        reclaimed = contact_maximum - confirm_steps

    available_rows = None
    if isinstance(maximum_action_steps, int) and not isinstance(
        maximum_action_steps, bool
    ):
        available_rows = (
            maximum_action_steps - settle - (planned_steps - reclaimed)
        )
        if available_rows < 4 + settle_rows:
            receipt["reason"] = "task_step_budget_insufficient"
            receipt["rows_required"] = 4 + settle_rows
            receipt["rows_available"] = available_rows
            return plan, receipt

    interp_rows = max(4, math.ceil(max_delta / feasible_step))
    budget_limited = False
    if available_rows is not None and interp_rows > available_rows - settle_rows:
        interp_rows = available_rows - settle_rows
        budget_limited = True
    if interp_rows > CONTACT_ENTRY_BRANCH_REPLAY_MAX_TOTAL_ROWS:
        interp_rows = CONTACT_ENTRY_BRANCH_REPLAY_MAX_TOTAL_ROWS
        budget_limited = True
    if reclaimed:
        contact_row["maximum_steps"] = confirm_steps

    # Every row commands the SAME solved posture.  The servo's slew and
    # per-joint feasible-lead bounds turn that into a ramp at exactly the rate
    # the slowest joint can follow, which is strictly better than an open-loop
    # interpolation: C33 hand-rolled the ramp, ran ahead of a lagging wrist,
    # and spent 37% of its rows saturated.  A closed-loop command cannot
    # outrun the joint, so the row count only has to be generous enough for
    # the traverse rather than exactly right.
    rows: list[dict[str, Any]] = [
        {
            "phase_id": CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID,
            "arm_joint_positions": [float(value) for value in end],
            "gripper_state": "open",
            "max_joint_delta_rad": feasible_step,
            "max_joint_setpoint_lead_rad": max(
                feasible_step, float(contact_row.get("max_joint_setpoint_lead_rad") or 0.0)
            ),
        }
        for _ in range(interp_rows + settle_rows)
    ]
    actions[contact_index:contact_index] = rows
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    receipt.update(
        {
            "status": "applied",
            "reason": None,
            "interpolation_rows": interp_rows,
            "settle_rows": settle_rows,
            "maximum_joint_delta_rad": max_delta,
            "per_row_joint_step_rad": max_delta / interp_rows,
            "actuator_feasible_step_rad": feasible_step,
            # A budget-limited replay steps faster than the actuator can
            # track, so it will clip.  Sealing that keeps the difference
            # between "the entry was infeasible" and "the entry was rushed"
            # visible in the receipt instead of inferred from the outcome.
            "budget_limited": budget_limited,
            "contact_phase_steps_reclaimed": reclaimed,
            "rewritten_control_plan_digest": plan["plan_digest"],
        }
    )
    return plan, receipt


def _contact_open_joint_margin(global_ik: Mapping[str, Any]) -> float | None:
    phases = global_ik.get("phases")
    if not isinstance(phases, list):
        return None
    for phase in phases:
        if not isinstance(phase, Mapping) or phase.get("phase_id") != "contact_open":
            continue
        selected = phase.get("selected")
        if not isinstance(selected, Mapping):
            return None
        try:
            margin = float(selected["minimum_joint_limit_margin_rad"])
        except (KeyError, TypeError, ValueError):
            return None
        return margin if math.isfinite(margin) else None
    return None


def _select_parallel_jaw_control_plan(
    *,
    servo: Any,
    control_plan: Mapping[str, Any],
    construction_bound_targets: Sequence[Mapping[str, Any]],
    reference_seeds: Sequence[Sequence[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Solve both equivalent jaw signs and select once, before physics moves."""

    nominal = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    equivalent = _parallel_jaw_equivalent_control_plan(control_plan)
    variants = []
    solved_payloads = []
    for variant_id, plan, bound_targets in (
        ("normalized_nominal", nominal, construction_bound_targets),
        ("parallel_jaw_equivalent", equivalent, []),
    ):
        targets, preflight = _control_plan_global_ik_joint_targets(
            servo=servo,
            control_plan=plan,
            bound_targets=bound_targets,
            reference_seeds=reference_seeds,
        )
        margin = _contact_open_joint_margin(preflight)
        admissible = (
            preflight.get("status") == "all_unique_poses_solved_or_bound"
            and margin is not None
            and margin >= 0.0
        )
        row = {
            "variant_id": variant_id,
            "control_plan_digest": plan.get("plan_digest"),
            "all_unique_poses_solved_or_bound": preflight.get("status")
            == "all_unique_poses_solved_or_bound",
            "contact_open_minimum_joint_limit_margin_rad": margin,
            "admissible": admissible,
            "global_ik_preflight": preflight,
        }
        variants.append(row)
        if admissible:
            solved_payloads.append((margin, variant_id, plan, targets, preflight))
    if not solved_payloads:
        raise RuntimeError(
            "native_task_controls_no_parallel_jaw_variant_with_contact_margin"
        )
    # Prefer joint-limit room; retain the normalized nominal on an exact tie.
    selected = max(
        solved_payloads,
        key=lambda row: (row[0], row[1] == "normalized_nominal"),
    )
    margin, variant_id, plan, targets, preflight = selected
    receipt = {
        "schema_version": "native_task_controls_parallel_jaw_selection.v1",
        "status": "selected_before_physics_motion",
        "source_control_plan_digest": control_plan.get("plan_digest"),
        "sealed_input_control_plan_digest": control_plan.get(
            "runtime_normalized_source_plan_digest"
        ),
        "selected_variant_id": variant_id,
        "selected_control_plan_digest": plan.get("plan_digest"),
        "selected_contact_open_minimum_joint_limit_margin_rad": margin,
        "selection_rule": (
            "all_phases_solved_then_maximise_contact_open_joint_limit_margin_"
            "then_prefer_normalized_nominal"
        ),
        "variants": variants,
        "provider_mutation_performed": False,
        "physics_steps_performed": 0,
        "claim_boundary": (
            "off_sim_multistart_branch_selection_only;native_controls_remain_"
            "the_arrival_contact_dynamics_and_task_outcome_authority"
        ),
    }
    return plan, targets, {**receipt, "selected_global_ik_preflight": preflight}


def _input_binding_mismatches(
    *,
    manifest: Mapping[str, Any],
    packet_receipt: Mapping[str, Any],
    scene_plan: Mapping[str, Any],
    construction: Mapping[str, Any],
    control_plan: Mapping[str, Any],
) -> list[str]:
    """Name every disagreeing input relation instead of one opaque blocker.

    Every relation below costs a full paid provider run when it fails, so the
    blocker has to say which one disagreed. Each pair is also required to be
    *bound* -- two absent fields are not an agreement, they are two missing
    digests, and comparing them with ``!=`` alone would admit an unbound cell.
    """

    phase_plan = construction.get("construction_phase_plan") or {}
    pairs = (
        (
            "packet_receipt_digest_vs_manifest",
            packet_receipt.get("receipt_digest"),
            manifest.get("packet_receipt_digest"),
        ),
        (
            "scene_plan_digest_vs_manifest",
            scene_plan.get("plan_digest"),
            manifest.get("arena_scene_plan_digest"),
        ),
        (
            "construction_result_digest_vs_control_plan_planner_receipt",
            construction.get("result_digest"),
            control_plan.get("planner_receipt_digest"),
        ),
        (
            "control_plan_construction_scene_plan_digest_vs_scene_plan",
            control_plan.get("construction_scene_plan_digest"),
            scene_plan.get("plan_digest"),
        ),
        (
            "control_plan_construction_clearance_plan_digest_vs_construction",
            control_plan.get("construction_clearance_plan_digest"),
            phase_plan.get("plan_digest"),
        ),
        (
            "control_plan_plan_digest_vs_recomputed_canonical_digest",
            control_plan.get("plan_digest"),
            _canonical_digest(control_plan, field="plan_digest"),
        ),
    )
    mismatched = [
        relation
        for relation, left, right in pairs
        if not _bound_digest(left) or not _bound_digest(right) or left != right
    ]
    task_kind = control_plan.get("task_kind")
    if task_kind is not None and task_kind != scene_plan.get("task_kind"):
        mismatched.append("control_plan_task_kind_vs_scene_plan_task_kind")
    return mismatched


def _to_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    value_module = type(value).__module__
    if value_module == "warp" or value_module.startswith("warp."):
        import warp as wp

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{value_module}.{type(value).__name__}")


class _RigidScoringEnvironment:
    """Overlay exact scoring-frame/contact readback on the shared episode seam."""

    def __init__(
        self,
        *,
        environment: Any,
        task_readback: Any,
        task_spec: Mapping[str, Any],
    ) -> None:
        if not callable(getattr(task_readback, "read_task_sample", None)):
            raise RuntimeError("native_task_controls_rigid_readback_missing")
        try:
            contact_threshold = float(task_spec["task_contact_minimum_force_n"])
            collision_threshold = float(
                task_spec["collision_failure_minimum_force_n"]
            )
            bounds = task_spec["workspace_position_bounds_world_m"]
            lower = [float(value) for value in bounds["minimum"]]
            upper = [float(value) for value in bounds["maximum"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_rigid_measurement_contract_invalid"
            ) from exc
        if (
            not all(
                math.isfinite(value)
                for value in [contact_threshold, collision_threshold, *lower, *upper]
            )
            or contact_threshold <= 0.0
            or collision_threshold <= 0.0
            or len(lower) != 3
            or len(upper) != 3
            or any(low >= high for low, high in zip(lower, upper, strict=True))
        ):
            raise RuntimeError(
                "native_task_controls_rigid_measurement_contract_invalid"
            )
        self._environment = environment
        self._task_readback = task_readback
        self._contact_threshold = contact_threshold
        self._collision_threshold = collision_threshold
        self._workspace_lower = lower
        self._workspace_upper = upper

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def read_object_sample(self) -> dict[str, Any]:
        base = self._environment.read_object_sample()
        native = self._task_readback.read_task_sample()
        if not isinstance(base, Mapping) or not isinstance(native, Mapping):
            raise RuntimeError("native_task_controls_rigid_sample_invalid")
        try:
            pose = [float(value) for value in native["task_scoring_pose_world"]]
            task_force = float(native["task_robot_contact_peak_force_n"])
            support_force = float(native["task_support_contact_peak_force_n"])
            scene_force = float(native["task_scene_collision_peak_force_n"])
            robot_force = float(native["robot_scene_contact_peak_force_n"])
            forbidden_robot_force = float(
                native["robot_task_forbidden_collision_peak_force_n"]
            )
            locked_joint_violation = native[
                "locked_joint_containment_violation"
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("native_task_controls_rigid_sample_invalid") from exc
        if len(pose) != 7 or not all(
            math.isfinite(value)
            for value in [
                *pose,
                task_force,
                support_force,
                scene_force,
                robot_force,
                forbidden_robot_force,
            ]
        ) or not isinstance(locked_joint_violation, bool):
            raise RuntimeError("native_task_controls_rigid_sample_invalid")
        sample = dict(base)
        sample.update(native)
        sample.update(
            {
                "task_object_pose_world": pose,
                "task_contact_active": task_force >= self._contact_threshold,
                "support_contact_active": support_force >= self._contact_threshold,
                "robot_collision_failure": max(robot_force, forbidden_robot_force)
                >= self._collision_threshold,
                "forbidden_robot_task_collision_failure": (
                    forbidden_robot_force >= self._collision_threshold
                ),
                "locked_joint_containment_violation": locked_joint_violation,
                "scene_collision_failure": scene_force
                >= self._collision_threshold,
                "containment_violation": any(
                    value < low or value > high
                    for low, value, high in zip(
                        self._workspace_lower, pose[:3], self._workspace_upper, strict=True
                    )
                ),
                "controls_measurement_authority": (
                    "native_scoring_frame_pose_filtered_contacts_and_shared_"
                    "gripper_calibration"
                ),
            }
        )
        return sample


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    output = output_root / RESULT_FILENAME
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "phase_reached": "start",
        "native_isaac_executed": False,
        "controls_qualified": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
    }
    simulation_app = None
    try:
        _announce("input_verification")
        manifest = _load_and_verify_manifest(runtime)
        inputs = _verified_runtime_inputs(runtime, manifest)
        packet = runtime / "native_task_packet"
        packet_receipt = json.loads(
            (packet / "native_task_arena_packet_receipt.v1.json").read_text(
                encoding="utf-8"
            )
        )
        scene_plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text(
                encoding="utf-8"
            )
        )
        construction = json.loads(
            inputs["native_task_arena_construction_result.v1.json"].read_text(
                encoding="utf-8"
            )
        )
        control_plan = json.loads(
            inputs["adp_task_control_plan.v1.json"].read_text(encoding="utf-8")
        )
        binding_mismatches = _input_binding_mismatches(
            manifest=manifest,
            packet_receipt=packet_receipt,
            scene_plan=scene_plan,
            construction=construction,
            control_plan=control_plan,
        )
        if binding_mismatches:
            raise RuntimeError(
                "native_task_controls_input_binding_mismatch:"
                + ",".join(sorted(binding_mismatches))
            )
        result["manifest_input_digest"] = manifest["input_digest"]
        result["implementation_commit"] = manifest["implementation_commit"]
        result["packet_receipt_digest"] = packet_receipt["receipt_digest"]
        result["scene_plan_digest"] = scene_plan["plan_digest"]
        result["construction_result_digest"] = construction["result_digest"]
        result["input_control_plan_digest"] = control_plan["plan_digest"]
        result["control_plan_digest"] = control_plan["plan_digest"]
        result["phase_reached"] = "inputs_verified"
        _announce("input_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch_receipt = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        result["isaaclab_launch"] = launch_receipt
        _announce("simulation_app", "completed")

        from blueprint_pipeline.native_task_arena_construction_worker import (
            _gripper_convention_probe,
            preflight_native_dependency_matrix,
        )

        _announce("dependency_matrix")
        dependency_matrix = preflight_native_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependency_matrix
        if not dependency_matrix["all_required_available"]:
            result["blockers"].extend(dependency_matrix["blockers"])
            raise RuntimeError("native_task_controls_dependency_preflight_failed")
        result["phase_reached"] = "dependencies_qualified"
        _announce("dependency_matrix", "completed")

        import torch

        from blueprint_pipeline.adp009d_control_episode import (
            run_task_neutral_controls,
        )
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
            PINK_GLOBAL_REFERENCE_SEEDS,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
            NativeRigidTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )

        _announce("preconstruction_device_binding")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["preconstruction_device_binding"] = preconstruction
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_arena_preconstruction_failed")
        _announce("preconstruction_device_binding", "completed")

        _announce("environment_build")
        built = build_native_task_arena_environment(
            scene_plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        device_readback = read_native_task_arena_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["device_readback"] = device_readback
        if not device_readback["passed"]:
            result["blockers"].extend(device_readback["blockers"])
            raise RuntimeError("native_task_arena_device_binding_failed")
        env = built.env
        seed = int(scene_plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        task_kind = str(scene_plan["task_kind"])
        graph_rigid = (
            task_kind == "rigid_pick_place"
            and scene_plan["task_spec"].get("schema_version") == "adp_task_spec.v2"
        )
        readback = None
        if task_kind == "articulated_open_close":
            readback = NativeArticulatedTaskArenaReadback(built)
        elif graph_rigid:
            readback = NativeRigidTaskArenaReadback(built)
        result["native_isaac_executed"] = True
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        _announce("gripper_convention")
        gripper = _gripper_convention_probe(
            env=env, robot=robot, seed=seed, torch=torch
        )
        result["gripper_convention"] = gripper
        result["blockers"].extend(gripper["blockers"])
        if gripper["status"] != "measured":
            raise RuntimeError("native_task_controls_gripper_convention_unresolved")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(
            env=env, robot=robot, gripper_convention=gripper
        )
        if task_kind == "articulated_open_close":
            # The task sampler's separation and retreat verdicts must use the
            # same measured physical-pad midpoint that the servo controls.
            # Keeping the earlier inner-finger body-origin midpoint here made
            # one sealed sample report two different gripper positions.
            readback = NativeArticulatedTaskArenaReadback(
                built,
                grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
            )
        # The same sealed-reset measurement the construction worker retains:
        # which frame the controlled body is actually in, read back from the
        # finger bodies rather than assumed from a convention.  Taken here,
        # before any control has moved the arm.
        result["gripper_frame_axis_readback"] = (
            servo.current_gripper_frame_axis_readback()
        )
        normalized_control_plan = _normalized_control_plan_for_execution(
            control_plan=control_plan,
            task_spec=scene_plan["task_spec"],
        )
        result["normalized_control_plan_digest"] = normalized_control_plan[
            "plan_digest"
        ]
        construction_joint_targets = _construction_global_ik_joint_targets(
            construction=construction,
            control_plan=normalized_control_plan,
        )
        result["construction_global_ik_seed_binding"] = {
            "schema_version": "native_task_controls_global_ik_seed_binding.v1",
            "status": "bound" if construction_joint_targets else "not_available",
            "target_count": len(construction_joint_targets),
            "targets": construction_joint_targets,
            "construction_result_digest": construction["result_digest"],
            "provider_mutation_performed": False,
            "claim_boundary": (
                "replays_only_joint_branches_selected_by_the_bound_construction_"
                "receipt;native_controls_remain_the_arrival_contact_and_task_"
                "outcome_authority"
            ),
        }
        _announce("controls_global_ik_preflight")
        effective_control_plan, scripted_pose_joint_targets, jaw_selection = (
            _select_parallel_jaw_control_plan(
                servo=servo,
                control_plan=normalized_control_plan,
                construction_bound_targets=construction_joint_targets,
                reference_seeds=PINK_GLOBAL_REFERENCE_SEEDS,
            )
        )
        effective_control_plan, branch_replay = _with_contact_entry_branch_replay(
            control_plan=effective_control_plan,
            scripted_pose_joint_targets=scripted_pose_joint_targets,
            task_spec=scene_plan["task_spec"],
            actuator_feasible_step_rad=(
                servo.actuator_feasible_joint_step_rad()
                if callable(
                    getattr(servo, "actuator_feasible_joint_step_rad", None)
                )
                else None
            ),
        )
        result["contact_entry_branch_replay"] = branch_replay
        controls_global_ik = jaw_selection["selected_global_ik_preflight"]
        result["control_plan_variant_selection"] = jaw_selection
        result["controls_global_ik_preflight"] = controls_global_ik
        result["control_plan_digest"] = effective_control_plan["plan_digest"]
        _announce(
            "controls_global_ik_preflight",
            (
                "completed"
                if controls_global_ik["status"]
                == "all_unique_poses_solved_or_bound"
                else "blocked"
            ),
        )
        episode_environment, environment_receipt = (
            build_native_task_episode_environment(
                built=built,
                gripper_convention=gripper,
                servo=servo,
                task_readback=readback,
                to_tensor=_to_tensor,
                scripted_pose_joint_targets=scripted_pose_joint_targets,
            )
        )
        if graph_rigid:
            episode_environment = _RigidScoringEnvironment(
                environment=episode_environment,
                task_readback=readback,
                task_spec=scene_plan["task_spec"],
            )
            environment_receipt["task_state_source"] = (
                "native_rigid_scoring_frame_and_filtered_contact_readback"
            )
        result["episode_environment"] = environment_receipt
        result["phase_reached"] = "episode_environment_bound"
        _announce("gripper_convention", "completed")

        # Measure the gain x posture surface before the controls run.  Thirty-
        # four runs varied the controller one hypothesis at a time while the
        # binding constraint sat in the actuator configuration, so this trades
        # seconds of simulator time for the sweep those runs should have been.
        # Diagnostic only: it restores the gains, gates nothing, and a runtime
        # that cannot retune reports that rather than failing the controls.
        _announce("contact_posture_actuator_sweep")
        try:
            from blueprint_pipeline.native_task_arena_actuator_sweep import (
                candidate_postures,
                run_actuator_posture_sweep,
            )

            contact_row = next(
                row
                for row in effective_control_plan["scripted_positive_actions"]
                if isinstance(row, Mapping)
                and str(row.get("phase_id") or "") == "contact_open"
            )
            sweep = run_actuator_posture_sweep(
                environment=episode_environment,
                robot=robot,
                arm_joint_ids=list(range(7)),
                target_position_world_m=contact_row["target_position_world_m"],
                postures=candidate_postures(
                    controls_global_ik, phase_id="contact_open"
                ),
                gripper_open_command=float(gripper["open_command"]),
                max_joint_delta_rad=float(contact_row["max_joint_delta_rad"]),
                max_joint_setpoint_lead_rad=float(
                    contact_row["max_joint_setpoint_lead_rad"]
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - a diagnostic never fails a run
            sweep = {
                "schema_version": "native_task_arena_actuator_posture_sweep.v1",
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
                "cells": [],
            }
        result["contact_posture_actuator_sweep"] = sweep

        # C36 localized the defect to a kinematic constant: at the solved
        # contact posture, across a tenfold stiffness range and with joint
        # tracking at 0.007 rad, the measured fingertip sat +13.0 mm off in a
        # single axis.  The solver hits its own target; its model of where the
        # fingertip is disagrees with PhysX.  So solve for the posture whose
        # *measured* fingertip reaches the sealed target, by folding each
        # measured residual back into the solver's target.  The arrival gate is
        # untouched -- this stops handing it a posture the model had wrong.
        try:
            from blueprint_pipeline.native_task_arena_actuator_sweep import (
                calibrate_posture_to_measured_target,
            )

            contact_quaternion = contact_row["target_quaternion_world_xyzw"]
            contact_tolerance = float(contact_row["arrival_tolerance_m"])
            contact_orientation_tolerance = float(
                contact_row.get("arrival_orientation_tolerance_rad") or 0.08
            )

            def _solve_contact(target_position, seed_joints):
                solved = servo.solve_grasp_target_multistart(
                    target_position_world_m=list(target_position),
                    target_grasp_frame_quaternion_world_xyzw=contact_quaternion,
                    preferred_seeds=[list(seed_joints)],
                    reference_joint_positions_rad=list(seed_joints),
                    position_tolerance_m=contact_tolerance,
                    orientation_tolerance_rad=contact_orientation_tolerance,
                    preferred_minimum_joint_limit_margin_rad=0.05,
                    required_minimum_joint_limit_margin_rad=(
                        CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD
                    ),
                )
                selected = (solved or {}).get("selected")
                if not isinstance(selected, Mapping):
                    return None
                return selected.get("joint_positions_rad")

            seed_posture = next(
                (
                    row["joint_positions_rad"]
                    for row in scripted_pose_joint_targets
                    if str(row.get("phase_id") or "") == "contact_open"
                ),
                None,
            )
            calibration = (
                calibrate_posture_to_measured_target(
                    environment=episode_environment,
                    solve=_solve_contact,
                    target_position_world_m=contact_row["target_position_world_m"],
                    seed_joint_positions_rad=seed_posture,
                    gripper_open_command=float(gripper["open_command"]),
                    max_joint_delta_rad=float(contact_row["max_joint_delta_rad"]),
                    max_joint_setpoint_lead_rad=float(
                        contact_row["max_joint_setpoint_lead_rad"]
                    ),
                    arrival_tolerance_m=contact_tolerance,
                )
                if seed_posture is not None
                else {"status": "unavailable", "reason": "contact_posture_unsolved"}
            )
        except BaseException as exc:  # noqa: BLE001 - a diagnostic never fails a run
            calibration = {
                "schema_version": "native_task_arena_measured_posture_calibration.v1",
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
                "iterations": [],
            }
        result["contact_posture_measured_calibration"] = calibration

        # Adopt the calibrated posture only when physics says it is closer than
        # the one the model produced; otherwise the run keeps what it had.
        best = (calibration or {}).get("best") or {}
        first = ((calibration or {}).get("iterations") or [{}])[0]
        if (
            isinstance(best.get("joint_positions_rad"), list)
            and isinstance(best.get("measured_distance_to_target_m"), float)
            and isinstance(first.get("measured_distance_to_target_m"), float)
            and best["measured_distance_to_target_m"]
            < first["measured_distance_to_target_m"]
        ):
            scripted_pose_joint_targets = [
                (
                    {**row, "joint_positions_rad": list(best["joint_positions_rad"])}
                    if str(row.get("phase_id") or "") == "contact_open"
                    else row
                )
                for row in scripted_pose_joint_targets
            ]
            result["contact_posture_calibration_adopted"] = True
        else:
            result["contact_posture_calibration_adopted"] = False
        _announce(
            "contact_posture_actuator_sweep",
            "completed" if sweep.get("status") == "measured" else "blocked",
        )

        _announce("required_controls")
        pair = run_task_neutral_controls(
            environment=episode_environment,
            task_spec=scene_plan["task_spec"],
            control_plan=effective_control_plan,
            gripper_open_command=float(gripper["open_command"]),
            gripper_closed_command=float(gripper["closed_command"]),
            output_dir=output_root / "controls",
        )
        result["control_pair"] = pair
        result["controls_qualified"] = pair["cell_admitted_for_policy_execution"]
        result["blockers"].extend(pair["policy_execution_blockers"])
        result["blockers"] = sorted(set(result["blockers"]))
        result["status"] = "completed" if not result["blockers"] else "blocked"
        result["phase_reached"] = "required_controls_complete"
        _announce(
            "required_controls",
            "completed" if result["controls_qualified"] else "blocked",
        )
    except BaseException as exc:  # noqa: BLE001 - retain every paid failure
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
        result["blockers"].append(
            f"native_task_controls_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )
        result["blockers"] = sorted(set(result["blockers"]))
        result["status"] = "blocked"
        _announce(str(result["phase_reached"]), "blocked")
    finally:
        result["completed_at_unix_ns"] = time.time_ns()
        _persist(output, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
