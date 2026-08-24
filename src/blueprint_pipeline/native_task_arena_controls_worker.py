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


from blueprint_pipeline.native_task_arena_branch_continuity import (
    select_continuous_branch_chain,
)
from blueprint_pipeline.native_franka_action_math import (
    controlled_body_pose_for_rigid_grasp_frame_target,
)


RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
RESULT_FILENAME = "native_task_arena_control_result.v1.json"
DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME = (
    "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1.json"
)
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
# Phases that execute the posture their preflight solved, instead of letting
# the Cartesian controller re-derive one.  Contact is where the tolerance is
# tight enough that re-deriving loses: elsewhere a 20 mm departure still lands
# inside a 20 mm gate.
HOLD_SOLVED_VECTOR_PHASE_IDS = frozenset({"contact_open", "contact_close"})
CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID = "contact_open_branch_replay"
# The measured anchor replaces the old open-loop replay and keeps its public
# phase identity.  Phase 3 is now a gated replay of a posture PhysX measured,
# not ninety-six copies of an off-sim contact posture followed by another
# unrelated entry phase.
MEASURED_CONTACT_ENTRY_PHASE_ID = CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
MEASURED_CONTACT_FRONTIER_PHASE_PREFIX = "measured_contact_frontier_"
MEASURED_CONTACT_FRONTIER_FRACTIONS = (0.75, 0.5, 0.25)
MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS = 45
# C53 reached the measured open standoff exactly, but its twelve-step close
# dwell moved the fingers only ~1.7 mm before the next arm phase began.  The
# retained task budget has room for 39 steps, which lets the same bounded arm
# command advance from the no-contact anchor to the authored grasp while the
# gripper closes.  This is a ceiling, not a required dwell: bilateral contact
# plus pose stability can terminate it early.
MEASURED_CONTACT_CLOSE_MAXIMUM_STEPS = 39
CONTACT_APPROACH_ANCHOR_DISTANCE_M = 0.04
MEASURED_CONTACT_STANDOFF_AXIS_MINIMUM_DOT = 0.999
MEASURED_CONTACT_STANDOFF_MAXIMUM_TRACKING_ERROR_RAD = 0.01

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


def _contact_close_sweep_minimum_force_n(
    *,
    contact_close_row: Mapping[str, Any],
    task_state_binding: Mapping[str, Any],
) -> float:
    """Use the same authoritative threshold as the scored contact gate.

    Before the measured frontier is compiled, the optional plan-row field can
    legitimately be ``None``.  The episode later fills it from the task-state
    binding, so reading the row here made the diagnostic sweep reject a value
    that the real gate already knew.  If a row does carry a value, require it
    to agree rather than silently measuring against a different threshold.
    """

    try:
        threshold = float(task_state_binding["task_contact_minimum_force_n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("native_task_controls_contact_force_invalid") from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError("native_task_controls_contact_force_invalid")
    row_value = contact_close_row.get(
        "bilateral_task_contact_minimum_force_n"
    )
    if row_value is not None:
        try:
            row_threshold = float(row_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "native_task_controls_contact_force_plan_invalid"
            ) from exc
        if not math.isfinite(row_threshold) or row_threshold != threshold:
            raise RuntimeError("native_task_controls_contact_force_mismatch")
    return threshold


def _solve_closed_contact_on_reference_branch(
    *,
    servo: Any,
    contact_close_row: Mapping[str, Any],
    target_position_world_m: Sequence[float],
    reference_joint_positions_rad: Sequence[float],
) -> list[float] | None:
    """Refine a measured close miss without changing IK branches.

    The general multistart scorer normally prefers any solution above its
    0.05 rad comfort margin before considering joint travel. That is useful
    in free space and wrong for a physical residual correction: C72 measured
    one close branch, then its correction jumped to a distant high-margin
    branch and turned a 14--22 mm miss into a 60--222 mm miss. Making the
    required safety floor the preferred floor makes the solver's next key --
    distance from this measured branch -- authoritative here.
    """

    required_margin = CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD
    solved = servo.solve_grasp_target_multistart(
        target_position_world_m=list(target_position_world_m),
        target_grasp_frame_quaternion_world_xyzw=contact_close_row[
            "target_quaternion_world_xyzw"
        ],
        preferred_seeds=[list(reference_joint_positions_rad)],
        reference_joint_positions_rad=list(reference_joint_positions_rad),
        position_tolerance_m=contact_close_row["arrival_tolerance_m"],
        orientation_tolerance_rad=(
            contact_close_row.get("arrival_orientation_tolerance_rad") or 0.08
        ),
        preferred_minimum_joint_limit_margin_rad=required_margin,
        required_minimum_joint_limit_margin_rad=required_margin,
    )
    selected = solved.get("selected") if isinstance(solved, Mapping) else None
    joints = (
        selected.get("joint_positions_rad")
        if isinstance(selected, Mapping)
        else None
    )
    if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
        return None
    try:
        values = [float(value) for value in joints]
    except (TypeError, ValueError):
        return None
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        return None
    lead = contact_close_row.get("max_joint_setpoint_lead_rad")
    if lead is not None:
        try:
            maximum_lead = float(lead)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(maximum_lead) or maximum_lead <= 0.0:
            return None
        if any(
            abs(value - reference) > maximum_lead
            for value, reference in zip(
                values, reference_joint_positions_rad, strict=True
            )
        ):
            return None
    return values


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


def _persist_progress(output: Path, progress: Mapping[str, Any]) -> None:
    """Atomically retain an interrupt-safe diagnostic checkpoint."""

    normalised = json.loads(json.dumps(dict(progress), default=str))
    normalised["result_digest"] = _canonical_digest(
        normalised, field="result_digest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(normalised, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _announce_contact_acquisition_cell(progress: Mapping[str, Any]) -> None:
    """Emit one compact timeout-harvestable numeric summary per cell."""

    cell = progress.get("last_cell")
    if not isinstance(cell, Mapping):
        return
    forces = cell.get("terminal_task_contact_pad_forces_n")
    if not isinstance(forces, Mapping):
        forces = {}

    def _number(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "na"
        return f"{number:.5g}" if math.isfinite(number) else "na"

    print(
        "BLUEPRINT_CONTACT_ACQUISITION_PROGRESS:CELL:"
        f"i={int(cell.get('cell_index') or 0)}:"
        f"a={_number(cell.get('approach_offset_m'))}:"
        f"j={_number(cell.get('jaw_offset_m'))}:"
        f"l={_number(cell.get('lateral_offset_m'))}:"
        f"ok={int(cell.get('admitted') is True)}:"
        f"b={int(cell.get('maximum_consecutive_bilateral_steps') or 0)}:"
        f"lf={_number(forces.get('left_inner_finger'))}:"
        f"rf={_number(forces.get('right_inner_finger'))}:"
        f"d={_number(cell.get('terminal_distance_to_candidate_target_m'))}:"
        f"o={_number(cell.get('terminal_orientation_error_rad'))}:"
        f"ad={_number(cell.get('terminal_distance_to_authored_target_m'))}:"
        "pj="
        f"{_number((cell.get('best_bilateral_force_evidence') or {}).get('opposed_jaw_force_min_n'))}:"
        "fa="
        f"{_number((cell.get('best_bilateral_force_evidence') or {}).get('same_direction_approach_force_min_n'))}",
        flush=True,
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
    if set(verified) not in (
        required,
        {*required, DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME},
    ):
        raise RuntimeError("native_task_controls_runtime_inputs_incomplete")
    return verified


def _downstream_diagnostic_request(
    inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate the optional immutable opt-in; absence is always default-off."""

    path = inputs.get(DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME)
    if path is None:
        return {
            "status": "not_requested",
            "enabled": False,
            "provider_mutation_performed": False,
        }
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "native_task_controls_downstream_diagnostic_request_invalid"
        ) from exc
    if (
        not isinstance(request, Mapping)
        or request.get("schema_version")
        != "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1"
        or request.get("enabled") is not True
        or request.get("development_only") is not True
        or request.get("qualification_effect") != "none"
        or request.get("request_digest")
        != _canonical_digest(request, field="request_digest")
    ):
        raise RuntimeError(
            "native_task_controls_downstream_diagnostic_request_invalid"
        )
    return {**dict(request), "status": "requested"}


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
    start_joint_positions = list(reference)
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
        # C43 measured the arm at contact with panda_joint5 on its hard stop
        # for a third of the phase and panda_joint6 at full effort; off-sim IK
        # confirms the authored contact orientation admits a best joint-limit
        # margin of 0.0000 rad, while the same position with the orientation
        # free admits 0.8916 rad.  Roll about the gripper's own approach axis
        # is a real freedom for a parallel jaw straddling a 1.23 mm rim with an
        # 85 mm opening, and off-sim it buys 0.62 rad at 0.92 mm of position
        # error.  Contact phases search it; every other phase is untouched, and
        # the authored orientation is always a candidate.
        # The grasp roll is a property of the plan, applied before this loop
        # runs, so every phase is solved against whatever orientation the plan
        # carries.  Rolling inside the solver instead left the rolled pose
        # uncommanded: the live differential-IK controller drives the plan's
        # quaternion, and a rolled pose sealed anywhere else survives only as a
        # null-space preference that cannot move the primary pose objective.
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

    # The greedy chain above commits each phase before it knows whether the
    # next one can be reached from where it landed.  C39 paid for that: with
    # solves scored on the gate's frame, approach's own best branch sat
    # 0.615 rad from contact's, and the bounded entry path -- built to
    # interpolate hundredths of a radian -- was handed a whole arm
    # reconfiguration.  Every admissible branch is already sealed under each
    # phase's `attempts`, so enumerate the combinations and keep the chain
    # that is actually traversable.  Pure arithmetic, off-sim, no GPU time.
    continuity = select_continuous_branch_chain(
        phases=[phase for phase in phases if isinstance(phase, Mapping)],
        required_margin_rad=CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD,
        start_joint_positions_rad=start_joint_positions,
        bounded_entry_phase_id="contact_open",
    )
    if continuity.get("status") == "selected":
        chain = continuity["selected_chain"]
        chain_phase_ids = continuity.get("chain_phase_ids") or []
        by_phase_id = {
            str(phase.get("phase_id") or ""): phase
            for phase in phases
            if isinstance(phase, Mapping)
        }
        if len(chain) == len(chain_phase_ids):
            for phase_id, chosen in zip(chain_phase_ids, chain, strict=True):
                joints = [
                    float(value) for value in chosen["joint_positions_rad"]
                ]
                for row in targets:
                    if str(row.get("phase_id") or "") == phase_id:
                        row["joint_positions_rad"] = joints
                phase = by_phase_id.get(phase_id)
                if isinstance(phase, Mapping) and isinstance(
                    phase.get("selected"), Mapping
                ):
                    phase["selected"] = {**phase["selected"], **chosen}
        else:
            continuity = {
                **continuity,
                "status": "unavailable",
                "reason": "chain_length_does_not_match_chosen_phase_count",
            }
    receipt = {
        "branch_continuity": continuity,
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


def _pink_global_reference_seeds():
    """The multistart seeds, imported where used to keep the seam narrow."""

    from blueprint_pipeline.native_franka_pose_servo import (
        PINK_GLOBAL_REFERENCE_SEEDS,
    )

    return PINK_GLOBAL_REFERENCE_SEEDS


def _with_selected_grasp_roll(
    *, servo: Any, control_plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose one grasp roll for the whole grasp and write it into the plan.

    Evaluated across every grasp-holding phase, not at contact entry alone, so
    a roll cannot be admitted on the strength of the one phase that was
    measured while a later phase sits below the floor.  Fails open with a typed
    receipt: a scene whose authored orientation is already holdable keeps
    exactly the plan it had.
    """

    from blueprint_pipeline.native_task_arena_grasp_roll import (
        DEFAULT_GRASP_HOLDING_PHASE_IDS,
        DEFAULT_GRASP_ROLL_CANDIDATES_RAD,
        DEFAULT_REQUIRED_MARGIN_RAD,
        GRASP_ROLL_SCHEMA_VERSION,
        derive_rolled_control_plan,
        select_grasp_roll,
    )

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    actions = plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        return plan, {
            "schema_version": GRASP_ROLL_SCHEMA_VERSION,
            "status": "not_applied",
            "reason": "scripted_positive_actions_invalid",
        }
    axis = getattr(servo, "grasp_approach_axis_body", None)
    axis_body = axis() if callable(axis) else None
    if axis_body is None:
        return plan, {
            "schema_version": GRASP_ROLL_SCHEMA_VERSION,
            "status": "not_applied",
            "reason": "grasp_approach_axis_unavailable",
        }

    seen: set[str] = set()
    holding: list[Mapping[str, Any]] = []
    for row in actions:
        if not isinstance(row, Mapping) or row.get("mode") != "ik_pose":
            continue
        phase_id = str(row.get("phase_id") or "")
        if phase_id in DEFAULT_GRASP_HOLDING_PHASE_IDS and phase_id not in seen:
            seen.add(phase_id)
            holding.append(row)

    reference = [float(value) for value in servo.read_arm_joint_positions()]

    def _solve(phase: Mapping[str, Any], quaternion: Sequence[float]):
        try:
            solved = servo.solve_grasp_target_multistart(
                target_position_world_m=phase["target_position_world_m"],
                target_grasp_frame_quaternion_world_xyzw=list(quaternion),
                preferred_seeds=[reference, *_pink_global_reference_seeds()],
                reference_joint_positions_rad=reference,
                position_tolerance_m=float(phase["arrival_tolerance_m"]),
                orientation_tolerance_rad=float(
                    phase.get("arrival_orientation_tolerance_rad") or 0.08
                ),
                preferred_minimum_joint_limit_margin_rad=(
                    DEFAULT_REQUIRED_MARGIN_RAD
                ),
                required_minimum_joint_limit_margin_rad=0.0,
            )
        except Exception:  # noqa: BLE001 - an unsolvable roll is not a failure
            return None
        return solved.get("selected") if isinstance(solved, Mapping) else None

    selection = select_grasp_roll(
        holding_phases=holding,
        approach_axis_body=axis_body,
        solve_phase=_solve,
        roll_candidates_rad=DEFAULT_GRASP_ROLL_CANDIDATES_RAD,
        required_margin_rad=DEFAULT_REQUIRED_MARGIN_RAD,
    )
    if selection.get("status") != "selected":
        return plan, {**selection, "status": "not_applied"}
    roll = float(selection["selected_roll_rad"])
    if roll == 0.0:
        # The authored orientation is holdable: keep the plan untouched.
        return plan, {**selection, "status": "not_applied", "reason": "authored_roll_admissible"}
    derived, applied = derive_rolled_control_plan(
        control_plan=plan,
        roll_rad=roll,
        approach_axis_body=axis_body,
        holding_phase_ids=DEFAULT_GRASP_HOLDING_PHASE_IDS,
    )
    return derived, {**selection, **applied}


def _with_held_solved_contact_vectors(
    *,
    control_plan: Mapping[str, Any],
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Give the contact phases the posture their own preflight solved.

    C42 and C43 measured the Cartesian controller re-deriving a posture from
    scratch and walking 0.19 to 0.53 rad away from the solved vector, onto one
    whose own forward kinematics sat 20 mm outside the arrival gate while the
    solved vector's sat 4.8 mm inside it.  Tracking was never the problem: the
    arm reached what it was told within 0.008 rad.  It was told the wrong
    thing, and the good answer was computed and discarded -- the same shape as
    the grasp roll that was sealed in a receipt and never commanded.

    The arrival gate is untouched.  It still measures the real fingertip
    against the sealed target, so a solved vector that does not put it there
    fails exactly as honestly as before.
    """

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    actions = plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        return plan, {
            "schema_version": "native_task_controls_held_solved_vectors.v1",
            "status": "not_applied",
            "reason": "scripted_positive_actions_invalid",
        }
    held_by_phase: dict[str, list[float]] = {}
    for row in scripted_pose_joint_targets:
        phase_id = str(row.get("phase_id") or "")
        joints = row.get("joint_positions_rad")
        if phase_id in HOLD_SOLVED_VECTOR_PHASE_IDS and isinstance(joints, list):
            if len(joints) == 7:
                held_by_phase[phase_id] = [float(value) for value in joints]
    applied: list[str] = []
    for raw in actions:
        if not isinstance(raw, Mapping) or raw.get("mode") != "ik_pose":
            continue
        held = held_by_phase.get(str(raw.get("phase_id") or ""))
        if held is None:
            continue
        raw["hold_solved_arm_joint_positions_rad"] = list(held)
        applied.append(str(raw.get("phase_id") or ""))
    if applied:
        plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    return plan, {
        "schema_version": "native_task_controls_held_solved_vectors.v1",
        "status": "applied" if applied else "not_applied",
        "held_phase_ids": sorted(set(applied)),
        "source_control_plan_digest": control_plan.get("plan_digest"),
        "derived_control_plan_digest": plan.get("plan_digest"),
        "claim_boundary": (
            "commands_the_posture_the_preflight_solved_for_the_same_pose;"
            "the_native_arrival_and_contact_gates_are_unchanged"
        ),
    }


def _with_live_physx_dls_contact_close(
    *,
    control_plan: Mapping[str, Any],
    preferred_posture_joint_positions_rad: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the controller that the reset-isolated close cell admitted.

    A compensated contact-close target is an open-frame IK construction: the
    solved hand starts behind the authored target and linkage closure moves the
    pads onto it. Live DLS instead controls the closed TCP directly, so it must
    receive the authored arrival target. It also must not retain the solved
    joint hold, which otherwise wins dispatch before DLS is called at all.
    """

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    try:
        preferred = [
            float(value) for value in preferred_posture_joint_positions_rad
        ]
    except (TypeError, ValueError):
        preferred = []
    actions = plan.get("scripted_positive_actions")
    if (
        len(preferred) != 7
        or not all(math.isfinite(value) for value in preferred)
        or not isinstance(actions, list)
    ):
        return plan, {
            "schema_version": "native_task_controls_physx_dls_close.v1",
            "status": "not_applied",
            "reason": "preferred_posture_or_plan_invalid",
        }
    applied = 0
    source_target = None
    for row in actions:
        if (
            not isinstance(row, dict)
            or row.get("mode") != "ik_pose"
            or str(row.get("phase_id") or "") != "contact_close"
        ):
            continue
        authored = row.get("arrival_target_position_world_m")
        if not isinstance(authored, list) or len(authored) != 3:
            return plan, {
                "schema_version": "native_task_controls_physx_dls_close.v1",
                "status": "not_applied",
                "reason": "authored_arrival_target_missing",
            }
        source_target = list(row.get("target_position_world_m") or [])
        row["target_position_world_m"] = [float(value) for value in authored]
        row["hold_solved_arm_joint_positions_rad"] = None
        row["physx_dls_preferred_posture_joint_positions_rad"] = list(
            preferred
        )
        applied += 1
    if applied != 1:
        return plan, {
            "schema_version": "native_task_controls_physx_dls_close.v1",
            "status": "not_applied",
            "reason": "contact_close_row_count_invalid",
        }
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    return plan, {
        "schema_version": "native_task_controls_physx_dls_close.v1",
        "status": "applied",
        "source_control_plan_digest": control_plan.get("plan_digest"),
        "derived_control_plan_digest": plan["plan_digest"],
        "compensated_solver_target_position_world_m": source_target,
        "authored_dls_target_position_world_m": list(authored),
        "preferred_posture_joint_positions_rad": preferred,
        "claim_boundary": (
            "executes_only_a_reset_isolated_physics_admitted_controller_cell;"
            "the_authored_tcp_orientation_bilateral_contact_and_task_outcome_"
            "gates_are_unchanged"
        ),
    }


def _with_closed_pad_midpoint_compensated_contact(
    *,
    control_plan: Mapping[str, Any],
    gripper_convention: Mapping[str, Any],
    current_controlled_body_pose_world: Sequence[float],
    current_grasp_frame_pose_world: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Author contact_close for the TCP the *closed* linkage actually has.

    C54 measured a 13.56 mm pad-midpoint translation while the Robotiq closed.
    The IK preflight solved the open-pad midpoint at the authored grasp and the
    episode then scored the moving, measured pad midpoint.  That made
    contact_close pose-identical to contact_open during preflight, so its
    solution was bound and later discarded when contact_open was moved to the
    measured standoff.  Give contact_close its own compensated pose before
    preflight: the globally solved open midpoint starts one linkage-travel
    behind the authored target, and closure brings the measured midpoint onto
    that target.  The native TCP and bilateral-contact gates stay unchanged.
    """

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    receipt: dict[str, Any] = {
        "schema_version": "native_task_controls_closed_pad_compensation.v1",
        "status": "not_applied",
        "reason": None,
        "source_control_plan_digest": plan.get("plan_digest"),
    }
    actions = plan.get("scripted_positive_actions")
    midpoint_by_command = gripper_convention.get(
        "pad_midpoint_controlled_body_m"
    )
    try:
        open_command = float(gripper_convention["open_command"])
        closed_command = float(gripper_convention["closed_command"])
        current_body_pose = [
            float(value) for value in current_controlled_body_pose_world
        ]
        current_grasp_pose = [
            float(value) for value in current_grasp_frame_pose_world
        ]
    except (KeyError, TypeError, ValueError):
        receipt["reason"] = "gripper_convention_or_frame_pose_invalid"
        return plan, receipt
    if not isinstance(actions, list) or not isinstance(midpoint_by_command, Mapping):
        receipt["reason"] = "plan_or_pad_midpoints_invalid"
        return plan, receipt

    def _midpoint(command: float) -> list[float] | None:
        for raw_key, raw_value in midpoint_by_command.items():
            try:
                if abs(float(raw_key) - command) <= 1.0e-9:
                    value = [float(component) for component in raw_value]
                    return value if len(value) == 3 else None
            except (TypeError, ValueError):
                continue
        return None

    open_midpoint = _midpoint(open_command)
    closed_midpoint = _midpoint(closed_command)
    if (
        open_midpoint is None
        or closed_midpoint is None
        or len(current_body_pose) != 7
        or len(current_grasp_pose) != 7
        or not all(math.isfinite(value) for value in (*current_body_pose, *current_grasp_pose))
    ):
        receipt["reason"] = "pad_midpoint_travel_unavailable"
        return plan, receipt
    delta_body = [
        closed_midpoint[index] - open_midpoint[index] for index in range(3)
    ]
    travel_m = math.dist(delta_body, [0.0, 0.0, 0.0])
    if not math.isfinite(travel_m) or not 1.0e-4 <= travel_m <= 0.05:
        receipt["reason"] = "pad_midpoint_travel_out_of_bounds"
        return plan, receipt

    original_targets: list[list[float]] = []
    compensated_targets: list[list[float]] = []
    for row in actions:
        if (
            not isinstance(row, dict)
            or row.get("mode") != "ik_pose"
            or str(row.get("phase_id") or "") != "contact_close"
        ):
            continue
        try:
            target = [float(value) for value in row["target_position_world_m"]]
            orientation = [
                float(value)
                for value in row["target_quaternion_world_xyzw"]
            ]
            _, target_body_orientation = (
                controlled_body_pose_for_rigid_grasp_frame_target(
                    current_body_position_world_m=current_body_pose[:3],
                    current_body_quaternion_world_xyzw=current_body_pose[3:7],
                    current_grasp_frame_position_world_m=current_grasp_pose[:3],
                    current_grasp_frame_quaternion_world_xyzw=current_grasp_pose[3:7],
                    target_grasp_frame_position_world_m=target,
                    target_grasp_frame_quaternion_world_xyzw=orientation,
                )
            )
            delta_world = _quaternion_axis_world_xyzw(
                target_body_orientation, delta_body
            )
        except (KeyError, TypeError, ValueError):
            receipt["reason"] = "contact_close_pose_invalid"
            return json.loads(json.dumps(dict(control_plan), allow_nan=False)), receipt
        compensated = [
            target[index] - delta_world[index]
            for index in range(3)
        ]
        # The compensated pose is a command-space target.  The sealed task
        # target remains the arrival authority.  C56 exposed the distinction:
        # moving this row without preserving the original arrival target moved
        # the finish line by the same 13.56 mm we were compensating.
        row["arrival_target_position_world_m"] = list(target)
        row["target_position_world_m"] = compensated
        row["hold_arm_joint_positions_during_gripper_transition"] = False
        original_targets.append(target)
        compensated_targets.append(compensated)
    if not compensated_targets:
        receipt["reason"] = "contact_close_missing"
        return plan, receipt

    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    receipt.update(
        {
            "status": "applied",
            "reason": None,
            "pad_midpoint_delta_controlled_body_m": delta_body,
            "pad_midpoint_travel_m": travel_m,
            "original_target_positions_world_m": original_targets,
            "compensated_target_positions_world_m": compensated_targets,
            "derived_control_plan_digest": plan["plan_digest"],
            "claim_boundary": (
                "compensates_only_measured_gripper_linkage_tcp_travel_before_"
                "global_ik;native_tcp_bilateral_contact_and_outcome_gates_"
                "remain_authoritative"
            ),
        }
    )
    return plan, receipt


def _with_measured_contact_frontier(
    *,
    control_plan: Mapping[str, Any],
    reachability_probe: Mapping[str, Any],
    reclaimed_contact_steps: int = 0,
    task_spec: Mapping[str, Any] | None = None,
    task_contact_minimum_force_n: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Promote the deepest measured collision-free *open-entry* standoff.

    The reachability probe is stronger evidence than another off-sim solve: it
    records the commanded posture, reached joints, and measured grasp frame in
    the same runtime as the episode. C51 measured the actual geometry boundary:
    the -32 mm cell was inside the native arrival gate with no task contact and
    0.0002 rad joint tracking error, while -30 mm was already in task contact.
    C53 then proved that preserving that collision-free standoff through close
    and swing was a category error: contact_close passed its arm-pose gate while
    the fingers remained open, and the door never moved. Keep the measured pose
    only for entry/contact_open, then close while advancing to the original
    authored grasp and preserve every original globally solved swing target.
    Native TCP, bilateral contact, containment, and task-outcome gates remain
    authoritative.
    """

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    receipt: dict[str, Any] = {
        "schema_version": "native_task_controls_measured_contact_frontier.v3",
        "status": "not_applied",
        "reason": None,
        "source_control_plan_digest": plan.get("plan_digest"),
        "claim_boundary": (
            "uses_the_closest_zero_contact_physx_measured_pose_only_as_the_"
            "open_entry_anchor_then_advances_to_the_authored_grasp_while_"
            "closing_and_requires_bilateral_task_contact;original_global_"
            "swing_targets_arrival_containment_and_outcome_gates_are_preserved"
        ),
    }
    actions = plan.get("scripted_positive_actions")
    cells = reachability_probe.get("cells")
    if not isinstance(actions, list) or not isinstance(cells, list):
        receipt["reason"] = "plan_or_probe_cells_invalid"
        return plan, receipt

    branch_replay_rows = [
        row
        for row in actions
        if isinstance(row, Mapping)
        and str(row.get("phase_id") or "") == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
    ]
    replaced_branch_replay_rows = len(branch_replay_rows)
    branch_replay_step_limits = []
    for row in branch_replay_rows:
        try:
            value = float(row["max_joint_delta_rad"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            branch_replay_step_limits.append(value)
    preserved_branch_replay_step_rad = (
        min(branch_replay_step_limits) if branch_replay_step_limits else None
    )
    contact_index = next(
        (
            index
            for index, row in enumerate(actions)
            if isinstance(row, Mapping)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == "contact_open"
        ),
        None,
    )
    if contact_index is None:
        receipt["reason"] = "contact_open_missing"
        return plan, receipt
    contact = actions[contact_index]
    try:
        entry_maximum_steps = max(
            MEASURED_CONTACT_ENTRY_MAXIMUM_STEPS,
            int(contact.get("maximum_steps") or 0),
            # C52 proved that the measured endpoint is not enough by itself:
            # collapsing a 96-row, 0.005 rad/step replay into a 45-step phase
            # that inherited contact_open's 0.1 rad slew hit the door on its
            # second step (68 N) and saturated joint 7. Preserve the traversal
            # time already sized from the live actuator before replacing the
            # old rows with the measured endpoint.
            replaced_branch_replay_rows,
        )
    except (TypeError, ValueError):
        receipt["reason"] = "contact_open_invalid"
        return plan, receipt
    try:
        target = [float(value) for value in contact["target_position_world_m"]]
        tolerance = float(contact["arrival_tolerance_m"])
        target_orientation = [
            float(value) for value in contact["target_quaternion_world_xyzw"]
        ]
        orientation_tolerance = float(
            contact.get("arrival_orientation_tolerance_rad") or math.inf
        )
    except (KeyError, TypeError, ValueError):
        receipt["reason"] = "contact_open_invalid"
        return plan, receipt

    clearance_axis = _quaternion_axis_world_xyzw(
        target_orientation, [0.0, 0.0, -1.0]
    )
    clearance_norm = math.dist(clearance_axis, [0.0, 0.0, 0.0])
    if not math.isfinite(clearance_norm) or clearance_norm <= 1.0e-12:
        receipt["reason"] = "contact_open_clearance_axis_invalid"
        return plan, receipt
    clearance_axis = [value / clearance_norm for value in clearance_axis]

    candidates: list[
        tuple[float, Mapping[str, Any], list[float], list[float], float, float]
    ] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("status") != "measured":
            continue
        try:
            offset = [float(value) for value in cell["offset_m"]]
            joints = [float(value) for value in cell["joint_positions_rad"]]
            measured_error = float(cell["measured_distance_to_requested_m"])
            measured_orientation = [
                float(value)
                for value in cell[
                    "measured_grasp_frame_orientation_world_xyzw"
                ]
            ]
            measured_orientation_error = _quaternion_angle_xyzw(
                measured_orientation, target_orientation
            )
            contact_steps = int(cell.get("contact_steps") or 0)
            tracking_error = float(cell["joint_tracking_error_rad"])
        except (KeyError, TypeError, ValueError):
            continue
        offset_distance = math.dist(offset, [0.0, 0.0, 0.0])
        axis_alignment = (
            sum(
                offset[axis] * clearance_axis[axis]
                for axis in range(3)
            )
            / offset_distance
            if offset_distance > 1.0e-12
            else -1.0
        )
        if (
            len(offset) != 3
            or len(joints) != 7
            or not all(math.isfinite(value) for value in [*offset, *joints])
            or not math.isfinite(measured_error)
            or measured_error > tolerance
            or measured_orientation_error > orientation_tolerance
            or contact_steps != 0
            or bool(cell.get("aborted_on_contact_force"))
            or not math.isfinite(tracking_error)
            or tracking_error
            > MEASURED_CONTACT_STANDOFF_MAXIMUM_TRACKING_ERROR_RAD
            or axis_alignment < MEASURED_CONTACT_STANDOFF_AXIS_MINIMUM_DOT
        ):
            continue
        candidates.append(
            (
                offset_distance,
                cell,
                offset,
                joints,
                tracking_error,
                axis_alignment,
            )
        )
    if not candidates:
        receipt["reason"] = "no_noncontact_probe_cell_inside_arrival_gate"
        return plan, receipt

    standoff, cell, offset, joints, tracking_error, axis_alignment = min(
        candidates, key=lambda item: item[0]
    )
    if replaced_branch_replay_rows:
        actions[:] = [
            row
            for row in actions
            if not (
                isinstance(row, Mapping)
                and str(row.get("phase_id") or "")
                == CONTACT_ENTRY_BRANCH_REPLAY_PHASE_ID
            )
        ]
        contact_index = next(
            index
            for index, row in enumerate(actions)
            if isinstance(row, Mapping)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == "contact_open"
        )
        contact = actions[contact_index]
        # The removed replay borrowed most of contact_open's settle budget.
        # The measured anchor consumes part of the rows we just removed, so
        # only return the remainder to contact_open.  Returning every borrowed
        # step *and* adding the anchor made C49 exceed the sealed task budget
        # before a single episode action could run.
        requested_restoration = max(0, int(reclaimed_contact_steps))
        restoration_capacity = max(
            0, replaced_branch_replay_rows - entry_maximum_steps
        )
        restored = min(requested_restoration, restoration_capacity)
        if restored:
            contact["maximum_steps"] = int(contact["maximum_steps"]) + restored
    else:
        restored = 0

    # The measured cell is a no-contact open anchor. Rewriting contact_close
    # and the swing poses to it discarded the exact global IK solutions for
    # their authored poses because pose-key dispatch could no longer match.
    # Rewrite only contact_open; all downstream poses stay byte-for-byte at
    # the targets their preflight solved.
    contact["target_position_world_m"] = [
        target[axis] + standoff * clearance_axis[axis] for axis in range(3)
    ]
    contact["hold_solved_arm_joint_positions_rad"] = list(joints)
    rewritten_phase_ids = ["contact_open"]
    bound_phase_ids = ["contact_open"]

    contact_close = next(
        (
            row
            for row in actions
            if isinstance(row, dict)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == "contact_close"
        ),
        None,
    )
    if contact_close is None:
        receipt["reason"] = "contact_close_missing"
        original = json.loads(json.dumps(dict(control_plan), allow_nan=False))
        return original, receipt
    try:
        threshold = float(task_contact_minimum_force_n)
    except (TypeError, ValueError):
        threshold = math.nan
    if not math.isfinite(threshold) or threshold <= 0.0:
        receipt["reason"] = "task_contact_minimum_force_invalid"
        original = json.loads(json.dumps(dict(control_plan), allow_nan=False))
        return original, receipt
    contact_close["hold_arm_joint_positions_during_gripper_transition"] = False
    contact_close["require_bilateral_task_contact"] = True
    contact_close["bilateral_task_contact_minimum_force_n"] = threshold

    # Consume only real spare task budget and never invalidate a smaller plan.
    close_budget_added = 0
    maximum_action_steps = (
        task_spec.get("maximum_action_steps")
        if isinstance(task_spec, Mapping)
        else None
    )
    settle_steps = (
        int(task_spec.get("settle_window_samples") or 0)
        if isinstance(task_spec, Mapping)
        else 0
    )
    if isinstance(maximum_action_steps, int) and not isinstance(
        maximum_action_steps, bool
    ):
        planned_steps = sum(
            int(row.get("maximum_steps") or 1)
            if isinstance(row, Mapping) and row.get("mode") == "ik_pose"
            else 1
            for row in actions
        )
        # The measured entry is inserted below, so reserve its budget before
        # assigning any genuinely spare steps to contact_close.
        spare_steps = max(
            0,
            maximum_action_steps
            - settle_steps
            - planned_steps
            - entry_maximum_steps,
        )
        old_close_steps = int(contact_close["maximum_steps"])
        desired_addition = max(
            0, MEASURED_CONTACT_CLOSE_MAXIMUM_STEPS - old_close_steps
        )
        close_budget_added = min(spare_steps, desired_addition)
        contact_close["maximum_steps"] = old_close_steps + close_budget_added

    contact_index = next(
        index
        for index, row in enumerate(actions)
        if isinstance(row, Mapping)
        and row.get("mode") == "ik_pose"
        and str(row.get("phase_id") or "") == "contact_open"
    )
    contact = actions[contact_index]
    entry = dict(contact)
    entry.update(
        {
            "phase_id": MEASURED_CONTACT_ENTRY_PHASE_ID,
            # Use the exact requested pose whose reached joints and TCP were
            # measured, rather than reconstructing it from a rounded distance.
            "target_position_world_m": [
                target[axis] + offset[axis] for axis in range(3)
            ],
            "hold_solved_arm_joint_positions_rad": list(joints),
            # Preserve the actuator-sized traversal budget when one existed;
            # otherwise replay at least the settle budget that established
            # this cell as measured-good.
            "maximum_steps": entry_maximum_steps,
        }
    )
    if preserved_branch_replay_step_rad is not None:
        entry["max_joint_delta_rad"] = min(
            float(entry["max_joint_delta_rad"]),
            preserved_branch_replay_step_rad,
        )
    # Keep one measured replay phase for the scoreboard/evidence boundary, then
    # hold the same proven standoff in contact_open instead of driving onward
    # to the collision-producing authored endpoint.
    actions[contact_index:contact_index] = [entry]
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    receipt.update(
        {
            "status": "applied",
            "reason": None,
            "probe_offset_m": offset,
            "probe_measured_error_m": float(
                cell["measured_distance_to_requested_m"]
            ),
            "probe_measured_orientation_error_rad": _quaternion_angle_xyzw(
                cell["measured_grasp_frame_orientation_world_xyzw"],
                target_orientation,
            ),
            "probe_joint_positions_rad": joints,
            "probe_joint_tracking_error_rad": tracking_error,
            "probe_clearance_axis_alignment_dot": axis_alignment,
            "promoted_standoff_m": standoff,
            "clearance_axis_body": [0.0, 0.0, -1.0],
            "rewritten_grasp_holding_phase_ids": rewritten_phase_ids,
            "measured_joint_vector_bound_phase_ids": bound_phase_ids,
            "replaced_branch_replay_rows": replaced_branch_replay_rows,
            "preserved_branch_replay_step_rad": (
                preserved_branch_replay_step_rad
            ),
            "measured_entry_maximum_steps": entry_maximum_steps,
            "contact_close_step_budget_added": close_budget_added,
            "contact_close_maximum_steps": int(contact_close["maximum_steps"]),
            "bilateral_task_contact_minimum_force_n": threshold,
            "restored_contact_steps": restored,
            "restoration_limited_by_action_budget": restored
            < max(0, int(reclaimed_contact_steps)),
            "frontier_phase_ids": [entry["phase_id"], *rewritten_phase_ids],
            "synthetic_frontier_rows_inserted": 0,
            "rewritten_control_plan_digest": plan["plan_digest"],
        }
    )
    return plan, receipt


def _quaternion_angle_xyzw(
    left: Sequence[float], right: Sequence[float]
) -> float:
    try:
        first = [float(value) for value in left]
        second = [float(value) for value in right]
    except (TypeError, ValueError):
        return math.inf
    if len(first) != 4 or len(second) != 4:
        return math.inf
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if (
        not all(math.isfinite(value) for value in [*first, *second])
        or min(norm_first, norm_second) <= 1.0e-12
    ):
        return math.inf
    dot = abs(
        sum(a * b for a, b in zip(first, second, strict=True))
        / (norm_first * norm_second)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _unit_direction(values: Sequence[float]) -> list[float] | None:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0e-9:
        return None
    return [value / norm for value in vector]


def _cross_direction(
    left: Sequence[float], right: Sequence[float]
) -> list[float] | None:
    return _unit_direction(
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    )


def _contact_acquisition_axes(
    *,
    control_plan: Mapping[str, Any],
    authored_open_target: Sequence[float],
    authored_close_target: Sequence[float],
    pad_centers: Mapping[str, Any] | None,
) -> tuple[list[float] | None, list[float] | None, list[float] | None]:
    """Resolve the scene-relative advance, jaw, and lateral search axes.

    Contact-open and contact-close ordinarily share one TCP pose because the
    gripper state is the only intended change between them.  Their difference
    is therefore not a reliable approach direction.  The authored
    approach-to-contact line is the authoritative geometric axis; retain the
    open-to-close difference only as a compatibility fallback for older plans
    that encoded a distinct close advance.
    """

    clear_side_offset = _contact_approach_anchor_offset(control_plan)
    approach_axis = (
        _unit_direction([-float(value) for value in clear_side_offset])
        if clear_side_offset is not None
        else None
    )
    if approach_axis is None:
        try:
            approach_axis = _unit_direction(
                [
                    float(authored_close_target[index])
                    - float(authored_open_target[index])
                    for index in range(3)
                ]
            )
        except (IndexError, TypeError, ValueError):
            approach_axis = None
    jaw_axis = None
    if isinstance(pad_centers, Mapping):
        try:
            jaw_axis = _unit_direction(
                [
                    float(pad_centers["left"][index])
                    - float(pad_centers["right"][index])
                    for index in range(3)
                ]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            jaw_axis = None
    lateral_axis = (
        _cross_direction(approach_axis, jaw_axis)
        if approach_axis is not None and jaw_axis is not None
        else None
    )
    return approach_axis, jaw_axis, lateral_axis


def _with_contact_acquisition_candidate(
    *, control_plan: Mapping[str, Any], sweep: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay a physics-qualified open advance, then close without arm motion."""

    plan = json.loads(json.dumps(dict(control_plan), allow_nan=False))
    receipt: dict[str, Any] = {
        "schema_version": "native_task_controls_contact_acquisition_adoption.v1",
        "status": "not_applied",
        "reason": None,
        "source_control_plan_digest": plan.get("plan_digest"),
    }
    best = sweep.get("best_cell")
    actions = plan.get("scripted_positive_actions")
    if not isinstance(best, Mapping) or best.get("admitted") is not True:
        receipt["reason"] = "no_physics_admitted_contact_acquisition_cell"
        return plan, receipt
    if best.get("authored_target_gate_passed") is not True:
        receipt["reason"] = "physics_admitted_cell_authored_gate_unproven"
        return plan, receipt
    if not isinstance(actions, list):
        receipt["reason"] = "scripted_positive_actions_invalid"
        return plan, receipt
    try:
        target = [
            float(value)
            for value in best["candidate_target_position_world_m"]
        ]
        command_target = [
            float(value)
            for value in best.get(
                "candidate_command_target_position_world_m", target
            )
        ]
        joints = [
            float(value)
            for value in best["reached_open_joint_positions_rad"]
        ]
    except (KeyError, TypeError, ValueError):
        receipt["reason"] = "admitted_cell_replay_values_invalid"
        return plan, receipt
    if (
        len(target) != 3
        or len(command_target) != 3
        or len(joints) != 7
        or not all(
            math.isfinite(value)
            for value in [*target, *command_target, *joints]
        )
    ):
        receipt["reason"] = "admitted_cell_replay_values_invalid"
        return plan, receipt

    contact_open = next(
        (
            row
            for row in actions
            if isinstance(row, dict)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == "contact_open"
        ),
        None,
    )
    contact_close = next(
        (
            row
            for row in actions
            if isinstance(row, dict)
            and row.get("mode") == "ik_pose"
            and str(row.get("phase_id") or "") == "contact_close"
        ),
        None,
    )
    if contact_open is None or contact_close is None:
        receipt["reason"] = "contact_phase_missing"
        return plan, receipt
    try:
        authored_arrival_target = [
            float(value)
            for value in contact_close.get(
                "arrival_target_position_world_m",
                contact_close["target_position_world_m"],
            )
        ]
    except (KeyError, TypeError, ValueError):
        receipt["reason"] = "authored_arrival_target_invalid"
        return plan, receipt
    if len(authored_arrival_target) != 3 or not all(
        math.isfinite(value) for value in authored_arrival_target
    ):
        receipt["reason"] = "authored_arrival_target_invalid"
        return plan, receipt

    # The preceding measured branch-replay row remains the known-clear anchor.
    # Contact-open now performs only the open-jaw advance the sweep qualified.
    contact_open["target_position_world_m"] = list(command_target)
    contact_open["arrival_target_position_world_m"] = list(command_target)
    contact_open["hold_solved_arm_joint_positions_rad"] = list(joints)
    contact_open["gripper_state"] = "open"

    # Close from the pose the episode itself physically reached, not from a
    # second IK posture.  The episode seam snapshots those joints on entry and
    # keeps its native TCP, orientation, and bilateral-contact gates intact.
    # The command may replay a nearby physics-qualified grasp candidate, but
    # the authoritative arrival target never moves with the search cell.
    contact_close["target_position_world_m"] = list(command_target)
    contact_close["arrival_target_position_world_m"] = list(
        authored_arrival_target
    )
    contact_close["target_quaternion_world_xyzw"] = list(
        contact_open["target_quaternion_world_xyzw"]
    )
    contact_close["hold_arm_joint_positions_during_gripper_transition"] = True
    contact_close["hold_solved_arm_joint_positions_rad"] = list(joints)
    plan["plan_digest"] = _canonical_digest(plan, field="plan_digest")
    receipt.update(
        {
            "status": "applied",
            "reason": None,
            "adopted_cell_index": int(best["cell_index"]),
            "adopted_target_position_world_m": target,
            "adopted_command_target_position_world_m": command_target,
            "authoritative_arrival_target_position_world_m": (
                authored_arrival_target
            ),
            "adopted_open_joint_positions_rad": joints,
            "adopted_offsets_m": {
                "approach": float(best["approach_offset_m"]),
                "jaw": float(best["jaw_offset_m"]),
                "lateral": float(best["lateral_offset_m"]),
            },
            "derived_control_plan_digest": plan["plan_digest"],
            "claim_boundary": (
                "replays_only_a_physics_admitted_open_advance_and_holds_the_"
                "episode_reached_arm_joints_while_closing;native_bilateral_"
                "contact_and_task_outcome_gates_remain_authoritative"
            ),
        }
    )
    return plan, receipt


def _contact_approach_anchor_offset(
    control_plan: Mapping[str, Any],
    *,
    distance_m: float = CONTACT_APPROACH_ANCHOR_DISTANCE_M,
) -> list[float] | None:
    """One clear-side probe derived from the authored approach line.

    The direction is task-authored rather than a world-axis constant.  The
    anchor stays no farther from contact than either 40 mm or the approach
    pose itself, so it cannot overshoot behind a shorter authored approach.
    """

    actions = control_plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        return None
    positions: dict[str, list[float]] = {}
    for row in actions:
        if not isinstance(row, Mapping) or row.get("mode") != "ik_pose":
            continue
        phase_id = str(row.get("phase_id") or "")
        if phase_id not in {"approach", "contact_open"}:
            continue
        try:
            position = [float(value) for value in row["target_position_world_m"]]
        except (KeyError, TypeError, ValueError):
            return None
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            return None
        positions[phase_id] = position
    if set(positions) != {"approach", "contact_open"}:
        return None
    clear_side = [
        approach - contact
        for approach, contact in zip(
            positions["approach"], positions["contact_open"], strict=True
        )
    ]
    norm = math.sqrt(sum(value * value for value in clear_side))
    try:
        requested = float(distance_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(norm) or norm <= 1.0e-9 or not math.isfinite(requested) or requested <= 0:
        return None
    distance = min(norm, requested)
    return [distance * value / norm for value in clear_side]


def _contact_frontier_offsets(
    anchor_offset_m: Sequence[float], *, sample_count: int = 21
) -> list[list[float]]:
    """Walk the authored approach line from a proven anchor to contact."""

    try:
        anchor = [float(value) for value in anchor_offset_m]
    except (TypeError, ValueError):
        return []
    if (
        len(anchor) != 3
        or not all(math.isfinite(value) for value in anchor)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 2
    ):
        return []
    return [
        [
            value * (sample_count - 1 - index) / (sample_count - 1)
            for value in anchor
        ]
        for index in range(sample_count)
    ]


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


def _fallback_contact_open_postures(
    jaw_selection: Mapping[str, Any],
    *,
    control_plan: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return every distinct terminal IK posture for live physics to measure.

    An unsolved off-sim attempt is not an executable solution.  It is still a
    useful reset-isolated physics hypothesis when the entire contact-open
    family narrowly misses the gate, as C79 did.  Keep this surface explicit,
    variant-labelled and deduplicated; only a later measured pose/collision
    gate may promote one posture into the deterministic episode.
    """

    from blueprint_pipeline.native_task_arena_actuator_sweep import (
        candidate_postures,
    )

    variants = jaw_selection.get("variants")
    if not isinstance(variants, list):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    variant_plans: dict[str, Mapping[str, Any]] = {}
    if control_plan is not None:
        variant_plans = {
            "normalized_nominal": control_plan,
            "parallel_jaw_equivalent": _parallel_jaw_equivalent_control_plan(
                control_plan
            ),
        }
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        variant_id = str(variant.get("variant_id") or "")
        preflight = variant.get("global_ik_preflight")
        if not variant_id or not isinstance(preflight, Mapping):
            continue
        variant_plan = variant_plans.get(variant_id)
        target_row = (
            next(
                (
                    row
                    for row in variant_plan.get("scripted_positive_actions") or []
                    if isinstance(row, Mapping)
                    and row.get("mode") == "ik_pose"
                    and str(row.get("phase_id") or "") == "contact_open"
                ),
                None,
            )
            if isinstance(variant_plan, Mapping)
            else None
        )
        for posture in candidate_postures(
            preflight,
            phase_id="contact_open",
            include_unsolved_attempts=True,
        ):
            joints = posture.get("joint_positions_rad")
            if not isinstance(joints, list) or len(joints) != 7:
                continue
            key = (
                variant_id,
                *(round(float(value), 6) for value in joints),
            )
            if key in seen:
                continue
            seen.add(key)
            row = {
                **posture,
                "variant_id": variant_id,
                "posture_source": "jaw_variant_global_ik",
            }
            if isinstance(target_row, Mapping):
                row.update(
                    {
                        "candidate_command_target_position_world_m": list(
                            target_row["target_position_world_m"]
                        ),
                        "candidate_command_target_quaternion_world_xyzw": list(
                            target_row["target_quaternion_world_xyzw"]
                        ),
                        "authoritative_target_position_world_m": list(
                            target_row.get(
                                "arrival_target_position_world_m",
                                target_row["target_position_world_m"],
                            )
                        ),
                        "authoritative_target_quaternion_world_xyzw": list(
                            target_row.get(
                                "arrival_target_quaternion_world_xyzw",
                                target_row["target_quaternion_world_xyzw"],
                            )
                        ),
                    }
                )
            rows.append(row)
    return rows


def _bounded_orientation_reference_seeds(
    *,
    control_plan: Mapping[str, Any],
    jaw_selection: Mapping[str, Any],
    sweep: Mapping[str, Any],
    maximum_seed_count: int = 12,
) -> list[list[float]]:
    """Prefer bound physical references, then the best current jaw branches."""

    candidates: list[Sequence[float]] = []
    for row in control_plan.get("scripted_positive_actions") or []:
        if not isinstance(row, Mapping):
            continue
        preferred = row.get("physx_dls_preferred_posture_joint_positions_rad")
        if isinstance(preferred, Sequence) and not isinstance(
            preferred, (str, bytes)
        ):
            candidates.append(preferred)
    cells = [
        cell
        for cell in sweep.get("cells") or []
        if isinstance(cell, Mapping)
        and isinstance(cell.get("measured_distance_to_target_m"), (int, float))
        and isinstance(cell.get("measured_orientation_error_rad"), (int, float))
    ]
    cells.sort(
        key=lambda cell: (
            float(cell["measured_distance_to_target_m"]) / 0.005
        )
        ** 2
        + (float(cell["measured_orientation_error_rad"]) / 0.08) ** 2
    )
    candidates.extend(
        cell["commanded_joint_positions_rad"]
        for cell in cells
        if isinstance(cell.get("commanded_joint_positions_rad"), list)
    )
    for variant in jaw_selection.get("variants") or []:
        if not isinstance(variant, Mapping):
            continue
        for row in variant.get("scripted_pose_joint_targets") or []:
            if (
                isinstance(row, Mapping)
                and str(row.get("phase_id") or "")
                in {"contact_open", "contact_close"}
                and isinstance(row.get("joint_positions_rad"), list)
            ):
                candidates.append(row["joint_positions_rad"])
    seeds: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for raw in candidates:
        try:
            seed = [float(value) for value in raw]
        except (TypeError, ValueError):
            continue
        key = tuple(round(value, 8) for value in seed)
        if (
            len(seed) != 7
            or not all(math.isfinite(value) for value in seed)
            or key in seen
        ):
            continue
        seen.add(key)
        seeds.append(seed)
        if len(seeds) >= int(maximum_seed_count):
            break
    return seeds


def _bounded_orientation_joint_targets(
    *,
    control_plan: Mapping[str, Any],
    admitted_cell: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind biased-solve joints to unchanged open/close plan authorities."""

    metadata = admitted_cell.get("bounded_orientation_candidate")
    if not isinstance(metadata, Mapping):
        return []
    actions = control_plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        raise RuntimeError("bounded_orientation_control_plan_invalid")
    rows = {
        str(row.get("phase_id") or ""): row
        for row in actions
        if isinstance(row, Mapping)
        and str(row.get("phase_id") or "")
        in {"contact_open", "contact_close"}
    }
    try:
        open_joints = [
            float(value) for value in admitted_cell["commanded_joint_positions_rad"]
        ]
        close_joints = [
            float(value) for value in metadata["close_joint_positions_rad"]
        ]
        targets = [
            {
                "phase_id": phase_id,
                "target_position_world_m": [
                    float(value) for value in rows[phase_id]["target_position_world_m"]
                ],
                "target_quaternion_world_xyzw": [
                    float(value)
                    for value in rows[phase_id]["target_quaternion_world_xyzw"]
                ],
                "joint_positions_rad": joints,
            }
            for phase_id, joints in (
                ("contact_open", open_joints),
                ("contact_close", close_joints),
            )
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("bounded_orientation_joint_targets_invalid") from exc
    if any(
        len(row["joint_positions_rad"]) != 7
        or len(row["target_position_world_m"]) != 3
        or len(row["target_quaternion_world_xyzw"]) != 4
        or not all(
            math.isfinite(value)
            for field in (
                "joint_positions_rad",
                "target_position_world_m",
                "target_quaternion_world_xyzw",
            )
            for value in row[field]
        )
        for row in targets
    ):
        raise RuntimeError("bounded_orientation_joint_targets_invalid")
    return targets


def _should_run_bounded_orientation_fallback(
    *,
    live_dls_contact_fallback: bool,
    admitted_open_cell: Mapping[str, Any] | None,
    sweep: Mapping[str, Any],
) -> bool:
    measured = {
        str(cell.get("variant_id") or "")
        for cell in sweep.get("cells") or []
        if isinstance(cell, Mapping)
    }
    return bool(
        live_dls_contact_fallback
        and admitted_open_cell is None
        and {
            "normalized_nominal",
            "parallel_jaw_equivalent",
        }.issubset(measured)
    )


def _dispatch_physics_admitted_jaw_variant(
    *,
    normalized_control_plan: Mapping[str, Any],
    selected_control_plan: Mapping[str, Any],
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]],
    controls_global_ik: Mapping[str, Any],
    jaw_selection: Mapping[str, Any],
    admitted_open_cell: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Dispatch the exact jaw convention whose physical cell passed."""

    previous_variant_id = str(
        jaw_selection.get("selected_variant_id") or "normalized_nominal"
    )
    adopted_variant_id = str(
        admitted_open_cell.get("variant_id")
        or previous_variant_id
    )
    variant_row = next(
        (
            row
            for row in jaw_selection.get("variants") or []
            if isinstance(row, Mapping)
            and row.get("variant_id") == adopted_variant_id
        ),
        None,
    )
    plan = dict(selected_control_plan)
    targets = [dict(row) for row in scripted_pose_joint_targets]
    preflight = dict(controls_global_ik)
    if adopted_variant_id != previous_variant_id:
        if (
            adopted_variant_id != "parallel_jaw_equivalent"
            or not isinstance(variant_row, Mapping)
        ):
            raise RuntimeError(
                "native_task_controls_physics_variant_dispatch_invalid"
            )
        plan = _parallel_jaw_equivalent_control_plan(
            normalized_control_plan
        )
        variant_targets = variant_row.get("scripted_pose_joint_targets")
        variant_preflight = variant_row.get("global_ik_preflight")
        if not isinstance(variant_targets, list):
            raise RuntimeError(
                "native_task_controls_physics_variant_targets_missing"
            )
        if not isinstance(variant_preflight, Mapping):
            raise RuntimeError(
                "native_task_controls_physics_variant_preflight_missing"
            )
        targets = [
            dict(row) for row in variant_targets if isinstance(row, Mapping)
        ]
        preflight = dict(variant_preflight)
    receipt = {
        "schema_version": (
            "native_task_controls_physics_variant_dispatch.v1"
        ),
        "status": "applied",
        "previous_variant_id": previous_variant_id,
        "adopted_variant_id": adopted_variant_id,
        "variant_switched": adopted_variant_id != previous_variant_id,
        "selected_variant_control_plan_digest": plan.get("plan_digest"),
        "claim_boundary": (
            "dispatches_only_the_same_parallel_jaw_variant_whose_reset_"
            "isolated_cell_passed_the_unchanged_native_pose_collision_joint_"
            "limit_and_zero_contact_gates;continuous_episode_remains_"
            "authoritative"
        ),
    }
    return plan, targets, preflight, receipt


def _synthetic_post_phase5_checkpoint(
    *,
    control_plan: Mapping[str, Any],
    global_ik: Mapping[str, Any],
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]],
    task_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Choose a traceable synthetic boundary for phases 6--11 diagnostics.

    Only a terminal contact-close hypothesis already inside the unchanged
    position and orientation gates may become a checkpoint.  Joint continuity
    ranks that admitted set; it can never rescue an out-of-gate pose.  This
    state remains an initialization proposal, never evidence that Phase 5
    succeeded.
    """

    actions = control_plan.get("scripted_positive_actions")
    if not isinstance(actions, list):
        return None
    rows = {
        str(row.get("phase_id") or ""): row
        for row in actions
        if isinstance(row, Mapping)
    }
    contact = rows.get("contact_close")
    first_downstream = rows.get("joint_path_01")
    if not isinstance(contact, Mapping) or not isinstance(
        first_downstream, Mapping
    ):
        return None
    try:
        contact_position_tolerance_m = float(contact["arrival_tolerance_m"])
        contact_orientation_tolerance_rad = float(
            contact["arrival_orientation_tolerance_rad"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(contact_position_tolerance_m)
        or contact_position_tolerance_m <= 0.0
        or not math.isfinite(contact_orientation_tolerance_rad)
        or contact_orientation_tolerance_rad <= 0.0
    ):
        return None

    def _joints(value: Any) -> list[float] | None:
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        if len(result) != 7 or not all(math.isfinite(item) for item in result):
            return None
        return result

    def _pose_key(row: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
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

    candidates: list[dict[str, Any]] = []

    def _append_candidate(
        joints: Any,
        *,
        source: str,
        position_error_m: Any = None,
        orientation_error_rad: Any = None,
        margin_rad: Any = None,
    ) -> None:
        values = _joints(joints)
        try:
            position_error = float(position_error_m)
            orientation_error = float(orientation_error_rad)
        except (TypeError, ValueError):
            return
        if (
            values is None
            or not math.isfinite(position_error)
            or not math.isfinite(orientation_error)
            or position_error > contact_position_tolerance_m
            or orientation_error > contact_orientation_tolerance_rad
        ):
            return
        key = tuple(round(value, 8) for value in values)
        if any(row["key"] == key for row in candidates):
            return
        candidates.append(
            {
                "key": key,
                "joint_positions_rad": values,
                "source": source,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "minimum_joint_limit_margin_rad": margin_rad,
            }
        )

    first_downstream_pose = _pose_key(first_downstream)
    first_downstream_joints = _joints(
        first_downstream.get("hold_solved_arm_joint_positions_rad")
    )
    for row in scripted_pose_joint_targets:
        if not isinstance(row, Mapping):
            continue
        phase_id = str(row.get("phase_id") or "")
        key = _pose_key(row)
        if phase_id == "joint_path_01" or (
            first_downstream_pose is not None and key == first_downstream_pose
        ):
            first_downstream_joints = _joints(row.get("joint_positions_rad"))

    from blueprint_pipeline.native_task_arena_actuator_sweep import (
        candidate_postures,
    )

    for posture in candidate_postures(
        global_ik,
        phase_id="contact_close",
        include_unsolved_attempts=True,
    ):
        _append_candidate(
            posture.get("joint_positions_rad"),
            source="global_ik_gate_qualified_terminal_contact_close",
            position_error_m=posture.get("offsim_position_error_m"),
            orientation_error_rad=posture.get("offsim_orientation_error_rad"),
            margin_rad=posture.get("minimum_joint_limit_margin_rad"),
        )
    if not candidates:
        return None

    def _finite_or_default(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    selected = min(
        candidates,
        key=lambda row: (
            max(
                (
                    abs(left - right)
                    for left, right in zip(
                        row["joint_positions_rad"],
                        first_downstream_joints,
                        strict=True,
                    )
                ),
                default=0.0,
            )
            if first_downstream_joints is not None
            else 0.0,
            _finite_or_default(row.get("position_error_m"), math.inf),
            _finite_or_default(row.get("orientation_error_rad"), math.inf),
            -_finite_or_default(
                row.get("minimum_joint_limit_margin_rad"), -math.inf
            ),
        ),
    )
    raw_task_joints = contact.get("expected_joint_positions")
    if not isinstance(raw_task_joints, Mapping) or not raw_task_joints:
        raw_task_joints = task_spec.get("joint_reset_positions_rad")
    if not isinstance(raw_task_joints, Mapping) or not raw_task_joints:
        return None
    try:
        task_joints = {
            str(name): float(value)
            for name, value in raw_task_joints.items()
        }
    except (TypeError, ValueError):
        return None
    if not all(
        name and math.isfinite(value) for name, value in task_joints.items()
    ):
        return None
    checkpoint: dict[str, Any] = {
        "schema_version": "adp_task_synthetic_post_phase5_checkpoint.v1",
        "source_phase_id": "contact_close",
        "arm_joint_positions_rad": selected["joint_positions_rad"],
        "task_joint_positions_rad": task_joints,
        "gripper_state": "closed",
        "phase5_qualified": False,
        "initialization_authority": (
            "runtime_derived_from_gate_qualified_offsim_contact_close"
        ),
        "selected_candidate_source": selected["source"],
        "candidate_count": len(candidates),
        "selection_rule": (
            "unchanged_contact_position_and_orientation_gates_then_minimum_"
            "max_joint_hop_to_joint_path_01_then_pose_error_then_joint_"
            "limit_margin"
        ),
        "contact_position_tolerance_m": contact_position_tolerance_m,
        "contact_orientation_tolerance_rad": (
            contact_orientation_tolerance_rad
        ),
        "selected_offsim_position_error_m": selected["position_error_m"],
        "selected_offsim_orientation_error_rad": selected[
            "orientation_error_rad"
        ],
        "checkpoint_digest": "",
    }
    checkpoint["checkpoint_digest"] = _canonical_digest(
        checkpoint, field="checkpoint_digest"
    )
    return checkpoint


def _run_requested_synthetic_post_phase5_downstream_diagnostic(
    *,
    episode_environment: Any,
    scene_plan: Mapping[str, Any],
    control_plan: Mapping[str, Any],
    global_ik: Mapping[str, Any],
    scripted_pose_joint_targets: Sequence[Mapping[str, Any]],
    gripper: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Run the opt-in continuous suffix probe and no other controls work."""

    checkpoint = _synthetic_post_phase5_checkpoint(
        control_plan=control_plan,
        global_ik=global_ik,
        scripted_pose_joint_targets=scripted_pose_joint_targets,
        task_spec=scene_plan["task_spec"],
    )
    if checkpoint is None:
        return {
            "schema_version": (
                "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
            ),
            "status": "unavailable",
            "reason": "no_contact_pose_candidate_inside_unchanged_gates",
            "phase5_qualified": False,
            "qualification_effect": "none",
            "physics_steps_performed": 0,
            "claim_boundary": (
                "diagnostic_refusal_only;does_not_qualify_phase5_any_"
                "downstream_phase_policy_admission_or_task_success"
            ),
        }
    try:
        from blueprint_pipeline.adp009d_control_episode import (
            run_synthetic_post_phase5_downstream_diagnostic,
        )

        return run_synthetic_post_phase5_downstream_diagnostic(
            environment=episode_environment,
            task_spec=scene_plan["task_spec"],
            control_plan=control_plan,
            checkpoint=checkpoint,
            gripper_open_command=float(gripper["open_command"]),
            gripper_closed_command=float(gripper["closed_command"]),
            output_dir=output_root / "downstream_continuous_diagnostic",
        )
    except BaseException as exc:  # noqa: BLE001 - retain diagnostic gap
        return {
            "schema_version": (
                "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
            ),
            "status": "unavailable",
            "reason": f"{type(exc).__name__}:{exc}",
            "phase5_qualified": False,
            "qualification_effect": "none",
            "physics_steps_performed": 0,
            "claim_boundary": (
                "diagnostic_gap_only;continuous_controls_and_all_"
                "qualification_gates_unchanged"
            ),
        }


def _physics_admitted_contact_open_cell(
    sweep: Mapping[str, Any],
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> dict[str, Any] | None:
    """Choose a collision-free physical cell inside the unchanged pose gate."""

    cells = sweep.get("cells")
    if not isinstance(cells, list):
        return None
    admitted: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        try:
            position_error = float(cell["measured_distance_to_target_m"])
            orientation_error = float(cell["measured_orientation_error_rad"])
            commanded = [
                float(value)
                for value in cell["commanded_joint_positions_rad"]
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            len(commanded) != 7
            or not all(
                math.isfinite(value)
                for value in [position_error, orientation_error, *commanded]
            )
            or position_error > float(position_tolerance_m)
            or orientation_error > float(orientation_tolerance_rad)
            or cell.get("joint_limit_violation") is not False
            or cell.get("robot_collision_failure") is not False
            or cell.get("scene_collision_failure") is not False
            or cell.get("task_contact_active") is not False
        ):
            continue
        admitted.append(
            {
                **dict(cell),
                "pose_gate_passed": True,
                "commanded_joint_positions_rad": commanded,
            }
        )
    return min(
        admitted,
        key=lambda cell: (
            -float(
                cell.get("minimum_joint_limit_margin_rad")
                if isinstance(
                    cell.get("minimum_joint_limit_margin_rad"),
                    (int, float),
                )
                else -math.inf
            ),
            (
                float(cell["measured_distance_to_target_m"])
                / float(position_tolerance_m)
            )
            ** 2
            + (
                float(cell["measured_orientation_error_rad"])
                / float(orientation_tolerance_rad)
            )
            ** 2,
            float(cell.get("joint_tracking_error_rad") or math.inf),
        ),
        default=None,
    )


def _phase_is_solved_or_bound(
    global_ik: Mapping[str, Any], *, phase_id: str
) -> bool:
    phases = global_ik.get("phases")
    if not isinstance(phases, list):
        return False
    for phase in phases:
        if not isinstance(phase, Mapping) or phase.get("phase_id") != phase_id:
            continue
        return (
            phase.get("status") == "reused_bound_pose_solution"
            or isinstance(phase.get("selected"), Mapping)
        )
    return False


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
    evaluated_payloads = []
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
        contact_open_solved_or_bound = _phase_is_solved_or_bound(
            preflight, phase_id="contact_open"
        )
        contact_close_solved_or_bound = _phase_is_solved_or_bound(
            preflight, phase_id="contact_close"
        )
        contact_phases_solved_or_bound = (
            contact_open_solved_or_bound and contact_close_solved_or_bound
        )
        admissible = (
            contact_open_solved_or_bound
            and margin is not None
            and margin >= CONTROLS_CONTACT_REQUIRED_JOINT_MARGIN_RAD
        )
        row = {
            "variant_id": variant_id,
            "control_plan_digest": plan.get("plan_digest"),
            "all_unique_poses_solved_or_bound": preflight.get("status")
            == "all_unique_poses_solved_or_bound",
            "contact_open_solved_or_bound": contact_open_solved_or_bound,
            "contact_close_solved_or_bound": contact_close_solved_or_bound,
            "contact_phases_solved_or_bound": contact_phases_solved_or_bound,
            "contact_open_minimum_joint_limit_margin_rad": margin,
            "admissible": admissible,
            "global_ik_preflight": preflight,
            "scripted_pose_joint_targets": targets,
        }
        variants.append(row)
        evaluated_payloads.append((variant_id, plan, targets, preflight))
        if admissible:
            solved_payloads.append((margin, variant_id, plan, targets, preflight))
    if not solved_payloads:
        # C78 proved that the off-sim PINK solve can reject contact_open even
        # though earlier PhysX episodes physically admitted that same authored
        # pose inside the 5 mm gate.  PINK is a seed generator, not the task
        # authority.  Preserve the authored jaw sign and let the live DLS
        # controller face every unchanged pose, joint, collision, and contact
        # gate instead of turning a missing optional seed into a veto.
        variant_id, plan, targets, preflight = evaluated_payloads[0]
        receipt = {
            "schema_version": "native_task_controls_parallel_jaw_selection.v1",
            "status": "selected_live_physx_dls_fallback_before_physics_motion",
            "source_control_plan_digest": control_plan.get("plan_digest"),
            "sealed_input_control_plan_digest": control_plan.get(
                "runtime_normalized_source_plan_digest"
            ),
            "selected_variant_id": variant_id,
            "selected_control_plan_digest": plan.get("plan_digest"),
            "selected_contact_open_minimum_joint_limit_margin_rad": None,
            "selection_rule": (
                "preserve_normalized_nominal_when_no_off_sim_contact_open_"
                "branch_clears_margin_then_require_live_physx_gates"
            ),
            "variants": variants,
            "provider_mutation_performed": False,
            "physics_steps_performed": 0,
            "claim_boundary": (
                "missing_off_sim_seed_only;does_not_admit_contact_open_or_"
                "contact_close;native_pose_joint_collision_bilateral_contact_"
                "and_task_outcome_gates_remain_authoritative"
            ),
        }
        return plan, targets, {
            **receipt,
            "selected_global_ik_preflight": preflight,
        }
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
            "contact_open_solved_or_bound_then_maximise_contact_open_joint_"
            "limit_margin_then_prefer_normalized_nominal;contact_close_may_"
            "remain_live_physx_dls"
        ),
        "variants": variants,
        "provider_mutation_performed": False,
        "physics_steps_performed": 0,
        "claim_boundary": (
            "off_sim_contact_open_branch_selection_only;contact_close_and_"
            "non_contact_poses_without_global_ik_remain_live_dls_controls;"
            "native_arrival_contact_dynamics_and_task_outcome_gates_remain_"
            "authoritative"
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
        downstream_diagnostic_request = _downstream_diagnostic_request(inputs)
        result["synthetic_post_phase5_downstream_diagnostic_request"] = (
            downstream_diagnostic_request
        )
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
        normalized_control_plan, closed_pad_compensation = (
            _with_closed_pad_midpoint_compensated_contact(
                control_plan=normalized_control_plan,
                gripper_convention=gripper,
                current_controlled_body_pose_world=servo.current_body_pose_world(),
                current_grasp_frame_pose_world=servo.current_grasp_frame_pose_world(),
            )
        )
        result["closed_pad_midpoint_compensation"] = closed_pad_compensation
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
        # Seal what the arm is actually held to, so an off-sim margin claim can
        # be checked against the robot that will execute it rather than against
        # a stock description of a robot that will not.
        limits_reader = getattr(servo, "joint_position_limits_rad", None)
        result["arm_joint_position_limits"] = (
            limits_reader() if callable(limits_reader) else None
        )

        _announce("contact_grasp_roll_selection")
        normalized_control_plan, grasp_roll = _with_selected_grasp_roll(
            servo=servo, control_plan=normalized_control_plan
        )
        result["contact_grasp_roll"] = grasp_roll
        _announce(
            "contact_grasp_roll_selection",
            "completed" if grasp_roll.get("status") == "applied" else "blocked",
        )

        _announce("controls_global_ik_preflight")
        selected_control_plan, scripted_pose_joint_targets, jaw_selection = (
            _select_parallel_jaw_control_plan(
                servo=servo,
                control_plan=normalized_control_plan,
                construction_bound_targets=construction_joint_targets,
                reference_seeds=PINK_GLOBAL_REFERENCE_SEEDS,
            )
        )
        actuator_feasible_step_vector = (
            servo.actuator_feasible_joint_step_rad()
            if callable(
                getattr(servo, "actuator_feasible_joint_step_rad", None)
            )
            else None
        )
        selected_control_plan, held_vectors = _with_held_solved_contact_vectors(
            control_plan=selected_control_plan,
            scripted_pose_joint_targets=scripted_pose_joint_targets,
        )
        result["held_solved_contact_vectors"] = held_vectors
        effective_control_plan, branch_replay = _with_contact_entry_branch_replay(
            control_plan=selected_control_plan,
            scripted_pose_joint_targets=scripted_pose_joint_targets,
            task_spec=scene_plan["task_spec"],
            actuator_feasible_step_rad=actuator_feasible_step_vector,
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
                if jaw_selection.get("status")
                in {
                    "selected_before_physics_motion",
                    "selected_live_physx_dls_fallback_before_physics_motion",
                }
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
                scripted_pose_phase_targets=[
                    row
                    for row in effective_control_plan[
                        "scripted_positive_actions"
                    ]
                    if isinstance(row, Mapping) and row.get("mode") == "ik_pose"
                ],
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

        # This immutable opt-in is a separate, development-only probe.  C74
        # already sealed the reset-isolated 134-cell matrix, so repeating that
        # matrix (or any Phase-4/5 sweep) would only delay the one new causal
        # question: can the unchanged phases 6--11 execute continuously from a
        # safe, gate-filtered synthetic post-Phase-5 boundary?  Terminate after
        # sealing this receipt so the probe can never drift into controls
        # qualification or add work to the ordinary controls lane.
        if downstream_diagnostic_request.get("enabled") is True:
            _announce("synthetic_post_phase5_downstream_diagnostic")
            downstream_diagnostic = (
                _run_requested_synthetic_post_phase5_downstream_diagnostic(
                    episode_environment=episode_environment,
                    scene_plan=scene_plan,
                    control_plan=effective_control_plan,
                    global_ik=controls_global_ik,
                    scripted_pose_joint_targets=(
                        scripted_pose_joint_targets
                    ),
                    gripper=gripper,
                    output_root=output_root,
                )
            )
            result["synthetic_post_phase5_downstream_diagnostic"] = (
                downstream_diagnostic
            )
            result["downstream_phase_posture_matrix"] = {
                "schema_version": (
                    "native_task_arena_downstream_phase_posture_matrix.v1"
                ),
                "status": "not_run",
                "reason": (
                    "separate_continuous_downstream_diagnostic_mode"
                ),
                "represented_configuration_count": 0,
                "executed_cell_count": 0,
                "phase_reports": [],
                "claim_boundary": "c74_matrix_not_repeated",
            }
            result.update(
                {
                    "controls_qualified": False,
                    "qualification_effect": "none",
                    "development_only": True,
                    "diagnostic_only": True,
                    "phase5_qualified": False,
                    "phase_reached": (
                        "synthetic_post_phase5_downstream_diagnostic_complete"
                    ),
                }
            )
            diagnostic_measured = (
                downstream_diagnostic.get("status") == "measured"
            )
            if diagnostic_measured:
                result["status"] = "diagnostic_completed"
            else:
                reason = str(
                    downstream_diagnostic.get("reason")
                    or downstream_diagnostic.get("status")
                    or "unknown"
                )
                result["blockers"].append(
                    "synthetic_post_phase5_downstream_diagnostic_failed:"
                    + reason
                )
                result["blockers"] = sorted(set(result["blockers"]))
                result["status"] = "blocked"
            _announce(
                "synthetic_post_phase5_downstream_diagnostic",
                "completed" if diagnostic_measured else "blocked",
            )
            return 0 if diagnostic_measured else 1

        # A missing off-sim contact-open seed is precisely the condition that
        # needs the live controller as authority.  Running the 134-cell
        # phases-6--11 diagnostic first would spend most of the retained-worker
        # window before testing that cause.  C74 already sealed the downstream
        # feasibility matrix; on this fallback path, test Phase 5 first.
        live_dls_contact_fallback = jaw_selection.get("status") == (
            "selected_live_physx_dls_fallback_before_physics_motion"
        )
        _announce("downstream_phase_posture_matrix")
        if live_dls_contact_fallback:
            downstream_phase_matrix = {
                "schema_version": (
                    "native_task_arena_downstream_phase_posture_matrix.v1"
                ),
                "status": "unavailable",
                "reason": "deferred_until_live_dls_contact_open_is_measured",
                "represented_configuration_count": 0,
                "executed_cell_count": 0,
                "phase_reports": [],
                "claim_boundary": (
                    "phase5_first_diagnostic_order_only;continuous_controls_"
                    "and_all_task_gates_unchanged;c74_retains_prior_"
                    "downstream_feasibility_evidence"
                ),
            }
        else:
            try:
                from blueprint_pipeline.native_task_arena_actuator_sweep import (
                    run_downstream_phase_posture_matrix,
                )

                downstream_progress_path = (
                    output_root
                    / "downstream_phase_posture_matrix.progress.v1.json"
                )

                def _downstream_phase_progress(
                    progress: Mapping[str, Any],
                ) -> None:
                    _persist_progress(downstream_progress_path, progress)
                    last_phase = progress.get("last_phase")
                    if not isinstance(last_phase, Mapping):
                        return
                    print(
                        "BLUEPRINT_DOWNSTREAM_PHASE_MATRIX_PROGRESS:"
                        f"phase={last_phase.get('phase_id')}:"
                        f"completed={progress.get('completed_phase_count')}/"
                        f"{progress.get('total_phase_count')}:"
                        f"represented={progress.get('represented_configuration_count')}:"
                        f"executed={progress.get('executed_cell_count')}:"
                        f"pose_gate={last_phase.get('pose_gate_cell_count', 0)}",
                        flush=True,
                    )

                downstream_phase_matrix = run_downstream_phase_posture_matrix(
                    environment=episode_environment,
                    robot=robot,
                    arm_joint_ids=list(range(7)),
                    control_plan=effective_control_plan,
                    global_ik=controls_global_ik,
                    gripper_open_command=float(gripper["open_command"]),
                    gripper_closed_command=float(gripper["closed_command"]),
                    progress_callback=_downstream_phase_progress,
                )
            except BaseException as exc:  # noqa: BLE001 - diagnostic only
                downstream_phase_matrix = {
                    "schema_version": (
                        "native_task_arena_downstream_phase_posture_matrix.v1"
                    ),
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "represented_configuration_count": 0,
                    "executed_cell_count": 0,
                    "phase_reports": [],
                    "claim_boundary": (
                        "diagnostic_gap_only;continuous_controls_unchanged"
                    ),
                }
        result["downstream_phase_posture_matrix"] = downstream_phase_matrix
        _announce(
            "downstream_phase_posture_matrix",
            (
                "completed"
                if downstream_phase_matrix.get("status") == "measured"
                else "blocked"
            ),
        )

        _announce("synthetic_post_phase5_downstream_diagnostic")
        downstream_diagnostic = {
            "schema_version": (
                "adp_task_synthetic_post_phase5_downstream_diagnostic.v1"
            ),
            "status": "not_requested",
            "reason": "immutable_bundle_request_absent",
            "phase5_qualified": False,
            "qualification_effect": "none",
            "physics_steps_performed": 0,
            "claim_boundary": (
                "default_off;ordinary_controls_execution_unchanged"
            ),
        }
        result["synthetic_post_phase5_downstream_diagnostic"] = (
            downstream_diagnostic
        )
        _announce(
            "synthetic_post_phase5_downstream_diagnostic",
            (
                "completed"
                if downstream_diagnostic.get("status") == "measured"
                else "blocked"
            ),
        )

        contact_row = next(
            row
            for row in effective_control_plan["scripted_positive_actions"]
            if isinstance(row, Mapping)
            and str(row.get("phase_id") or "") == "contact_open"
        )
        contact_quaternion = contact_row["target_quaternion_world_xyzw"]
        contact_authoritative_position = contact_row.get(
            "arrival_target_position_world_m",
            contact_row["target_position_world_m"],
        )
        contact_authoritative_quaternion = contact_row.get(
            "arrival_target_quaternion_world_xyzw",
            contact_quaternion,
        )
        contact_tolerance = float(contact_row["arrival_tolerance_m"])
        contact_orientation_tolerance = float(
            contact_row.get("arrival_orientation_tolerance_rad") or 0.08
        )
        from blueprint_pipeline.native_task_arena_actuator_sweep import (
            candidate_postures,
        )

        contact_open_postures = candidate_postures(
            controls_global_ik, phase_id="contact_open"
        )
        if live_dls_contact_fallback:
            # C82's fallback carried terminal postures for both physically
            # equivalent jaw signs, but the matrix scored every one against
            # the selected nominal quaternion.  A parallel-jaw posture was
            # therefore guaranteed to report pi radians and could never be
            # adopted.  Measure those already-computed variants against their
            # own unchanged plan targets before searching any new bias shell.
            contact_open_postures = _fallback_contact_open_postures(
                jaw_selection,
                control_plan=normalized_control_plan,
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

        # First ask the cheapest question that can unlock execution: can this
        # runtime reproduce one clear-side pose on the authored approach line?
        # C46 already answered yes at 40 mm.  Re-measuring that one cell binds
        # the new run without replaying a 25-cell gain surface and a nine-cell
        # Cartesian cross whose result is already known.
        anchor_offset = _contact_approach_anchor_offset(effective_control_plan)
        try:
            from blueprint_pipeline.native_task_arena_actuator_sweep import (
                probe_target_reachability,
            )

            anchor_probe = (
                probe_target_reachability(
                    environment=episode_environment,
                    solve=_solve_contact,
                    base_target_position_world_m=contact_row[
                        "target_position_world_m"
                    ],
                    seed_joint_positions_rad=seed_posture,
                    gripper_open_command=float(gripper["open_command"]),
                    max_joint_delta_rad=float(contact_row["max_joint_delta_rad"]),
                    max_joint_setpoint_lead_rad=float(
                        contact_row["max_joint_setpoint_lead_rad"]
                    ),
                    offsets_m=[anchor_offset],
                )
                if seed_posture is not None and anchor_offset is not None
                else {
                    "schema_version": (
                        "native_task_arena_target_reachability_probe.v1"
                    ),
                    "status": "unavailable",
                    "reason": "contact_approach_anchor_unresolved",
                    "cells": [],
                }
            )
        except BaseException as exc:  # noqa: BLE001 - fall back to full diagnostics
            anchor_probe = {
                "schema_version": "native_task_arena_target_reachability_probe.v1",
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
                "cells": [],
            }
        anchor_cell = next(
            (
                cell
                for cell in anchor_probe.get("cells") or []
                if isinstance(cell, Mapping) and cell.get("status") == "measured"
            ),
            None,
        )
        anchor_orientation_error = (
            _quaternion_angle_xyzw(
                anchor_cell.get(
                    "measured_grasp_frame_orientation_world_xyzw"
                ),
                contact_quaternion,
            )
            if isinstance(anchor_cell, Mapping)
            else math.inf
        )
        anchor_admitted = bool(
            isinstance(anchor_cell, Mapping)
            and isinstance(
                anchor_cell.get("measured_distance_to_requested_m"), (int, float)
            )
            and float(anchor_cell["measured_distance_to_requested_m"])
            <= contact_tolerance
            and anchor_orientation_error <= contact_orientation_tolerance
            and int(anchor_cell.get("contact_steps") or 0) == 0
        )
        result["contact_approach_anchor_probe"] = {
            **anchor_probe,
            "anchor_offset_m": anchor_offset,
            "anchor_orientation_error_rad": anchor_orientation_error,
            "admitted_for_short_path": anchor_admitted,
        }

        # Measure the full gain x posture surface only when the one-cell anchor
        # fails.  The fallback retains the earlier diagnostic power without
        # charging every successful run for it.
        _announce("contact_posture_actuator_sweep")
        if anchor_admitted:
            sweep = {
                "schema_version": "native_task_arena_actuator_posture_sweep.v1",
                "status": "skipped",
                "reason": "measured_approach_anchor_inside_arrival_gate",
                "cells": [],
            }
        else:
            try:
                from blueprint_pipeline.native_task_arena_actuator_sweep import (
                    DEFAULT_WRIST_GAIN_CANDIDATES,
                    run_actuator_posture_sweep,
                )

                contact_open_progress_path = (
                    output_root
                    / "contact_open_posture_matrix.progress.v1.json"
                )

                def _contact_open_progress(
                    progress: Mapping[str, Any],
                ) -> None:
                    _persist_progress(contact_open_progress_path, progress)
                    completed = int(
                        progress.get("completed_cell_count") or 0
                    )
                    total = int(progress.get("total_cell_count") or 0)
                    last = progress.get("last_cell")
                    if completed % 10 == 0 or completed == total:
                        print(
                            "BLUEPRINT_CONTACT_OPEN_MATRIX_PROGRESS:"
                            f"completed={completed}/{total}:"
                            f"position_m={(last or {}).get('measured_distance_to_target_m')}:"
                            f"orientation_rad={(last or {}).get('measured_orientation_error_rad')}",
                            flush=True,
                        )

                sweep = run_actuator_posture_sweep(
                    environment=episode_environment,
                    robot=robot,
                    arm_joint_ids=list(range(7)),
                    target_position_world_m=contact_authoritative_position,
                    target_orientation_world_xyzw=(
                        contact_authoritative_quaternion
                    ),
                    postures=contact_open_postures,
                    gripper_open_command=float(gripper["open_command"]),
                    max_joint_delta_rad=float(contact_row["max_joint_delta_rad"]),
                    max_joint_setpoint_lead_rad=float(
                        contact_row["max_joint_setpoint_lead_rad"]
                    ),
                    wrist_gain_candidates=(
                        ((400.0, 80.0),)
                        if live_dls_contact_fallback
                        else DEFAULT_WRIST_GAIN_CANDIDATES
                    ),
                    progress_callback=_contact_open_progress,
                )
            except BaseException as exc:  # noqa: BLE001 - diagnostic only
                sweep = {
                    "schema_version": "native_task_arena_actuator_posture_sweep.v1",
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "cells": [],
                }
        result["contact_open_jaw_variant_posture_sweep"] = sweep

        result["contact_posture_actuator_sweep"] = sweep

        admitted_open_cell = _physics_admitted_contact_open_cell(
            sweep,
            position_tolerance_m=contact_tolerance,
            orientation_tolerance_rad=contact_orientation_tolerance,
        )
        bounded_orientation_report: dict[str, Any] = {
            "schema_version": (
                "native_task_arena_bounded_orientation_search.v1"
            ),
            "status": "not_attempted",
            "reason": (
                "jaw_variant_cell_admitted"
                if admitted_open_cell is not None
                else "both_jaw_variants_not_measured"
            ),
            "represented_candidate_count": 0,
            "solved_candidate_count": 0,
            "executed_cell_count": 0,
        }
        bounded_sweep: dict[str, Any] | None = None
        if _should_run_bounded_orientation_fallback(
            live_dls_contact_fallback=live_dls_contact_fallback,
            admitted_open_cell=admitted_open_cell,
            sweep=sweep,
        ):
            from blueprint_pipeline.native_task_arena_bounded_orientation import (
                build_bounded_orientation_postures,
            )

            bounded_reference_seeds = _bounded_orientation_reference_seeds(
                control_plan=normalized_control_plan,
                jaw_selection=jaw_selection,
                sweep=sweep,
            )
            if not bounded_reference_seeds:
                bounded_reference_seeds = [
                    [float(value) for value in servo.read_arm_joint_positions()]
                ]

            def _solve_bounded_orientation(
                phase_id,
                target_position,
                target_quaternion,
                preferred_seeds,
            ):
                del phase_id
                reference = list(preferred_seeds[0])
                solved = servo.solve_grasp_target_multistart(
                    target_position_world_m=list(target_position),
                    target_grasp_frame_quaternion_world_xyzw=list(
                        target_quaternion
                    ),
                    preferred_seeds=[list(seed) for seed in preferred_seeds],
                    reference_joint_positions_rad=reference,
                    position_tolerance_m=contact_tolerance,
                    orientation_tolerance_rad=contact_orientation_tolerance,
                    preferred_minimum_joint_limit_margin_rad=0.05,
                    required_minimum_joint_limit_margin_rad=0.0,
                )
                selected = (
                    solved.get("selected")
                    if isinstance(solved, Mapping)
                    else None
                )
                return dict(selected) if isinstance(selected, Mapping) else None

            bounded_postures, bounded_orientation_report = (
                build_bounded_orientation_postures(
                    variant_plans=(
                        ("normalized_nominal", normalized_control_plan),
                        (
                            "parallel_jaw_equivalent",
                            _parallel_jaw_equivalent_control_plan(
                                normalized_control_plan
                            ),
                        ),
                    ),
                    solve_phase=_solve_bounded_orientation,
                    reference_joint_seeds=bounded_reference_seeds,
                )
            )
            if bounded_postures:
                try:
                    from blueprint_pipeline.native_task_arena_actuator_sweep import (
                        run_actuator_posture_sweep,
                    )

                    bounded_progress_path = (
                        output_root
                        / "contact_open_bounded_orientation.progress.v1.json"
                    )

                    def _bounded_orientation_progress(
                        progress: Mapping[str, Any],
                    ) -> None:
                        _persist_progress(bounded_progress_path, progress)
                        completed = int(
                            progress.get("completed_cell_count") or 0
                        )
                        total = int(progress.get("total_cell_count") or 0)
                        if completed % 5 == 0 or completed == total:
                            last = progress.get("last_cell") or {}
                            print(
                                "BLUEPRINT_BOUNDED_ORIENTATION_PROGRESS:"
                                f"completed={completed}/{total}:"
                                "position_m="
                                f"{last.get('measured_distance_to_target_m')}:"
                                "orientation_rad="
                                f"{last.get('measured_orientation_error_rad')}",
                                flush=True,
                            )

                    bounded_sweep = run_actuator_posture_sweep(
                        environment=episode_environment,
                        robot=robot,
                        arm_joint_ids=list(range(7)),
                        target_position_world_m=contact_authoritative_position,
                        target_orientation_world_xyzw=(
                            contact_authoritative_quaternion
                        ),
                        postures=bounded_postures,
                        gripper_open_command=float(gripper["open_command"]),
                        max_joint_delta_rad=float(
                            contact_row["max_joint_delta_rad"]
                        ),
                        max_joint_setpoint_lead_rad=float(
                            contact_row["max_joint_setpoint_lead_rad"]
                        ),
                        wrist_gain_candidates=((400.0, 80.0),),
                        progress_callback=_bounded_orientation_progress,
                    )
                except BaseException as exc:  # noqa: BLE001 - diagnostic only
                    bounded_sweep = {
                        "schema_version": (
                            "native_task_arena_actuator_posture_sweep.v1"
                        ),
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "cells": [],
                    }
                bounded_orientation_report["executed_cell_count"] = len(
                    bounded_sweep.get("cells") or []
                )
                admitted_open_cell = _physics_admitted_contact_open_cell(
                    bounded_sweep,
                    position_tolerance_m=contact_tolerance,
                    orientation_tolerance_rad=contact_orientation_tolerance,
                )
                bounded_orientation_report["status"] = (
                    "physics_cell_admitted"
                    if admitted_open_cell is not None
                    else "physics_measured_no_admission"
                )
                bounded_orientation_report["reason"] = (
                    None
                    if admitted_open_cell is not None
                    else "no_reset_isolated_cell_passed_unchanged_gates"
                )
                bounded_orientation_report["physics_admitted_cell"] = (
                    admitted_open_cell
                )
        result["bounded_contact_orientation_search"] = (
            bounded_orientation_report
        )
        if bounded_sweep is not None:
            result["contact_open_bounded_orientation_sweep"] = bounded_sweep
        result["contact_open_physics_adoption"] = {
            "schema_version": (
                "native_task_controls_contact_open_physics_adoption.v1"
            ),
            "status": "applied" if admitted_open_cell is not None else "not_applied",
            "measured_cell": admitted_open_cell,
            "represented_posture_count": (
                len(contact_open_postures)
                + int(bounded_orientation_report.get("solved_candidate_count") or 0)
            ),
            "executed_cell_count": (
                len(sweep.get("cells") or [])
                + int(bounded_orientation_report.get("executed_cell_count") or 0)
            ),
            "claim_boundary": (
                "promotes_only_a_reset_isolated_cell_that_passed_the_unchanged_"
                "position_orientation_joint_limit_collision_and_zero_contact_"
                "gates;the_continuous_episode_remains_authoritative"
            ),
        }
        if admitted_open_cell is not None:
            (
                selected_control_plan,
                scripted_pose_joint_targets,
                controls_global_ik,
                variant_dispatch,
            ) = _dispatch_physics_admitted_jaw_variant(
                normalized_control_plan=normalized_control_plan,
                selected_control_plan=selected_control_plan,
                scripted_pose_joint_targets=scripted_pose_joint_targets,
                controls_global_ik=controls_global_ik,
                jaw_selection=jaw_selection,
                admitted_open_cell=admitted_open_cell,
            )
            result["contact_open_physics_variant_dispatch"] = variant_dispatch
            adopted_open = list(
                admitted_open_cell["commanded_joint_positions_rad"]
            )
            adopted_target_position = list(
                admitted_open_cell.get(
                    "candidate_command_target_position_world_m"
                )
                or contact_row["target_position_world_m"]
            )
            adopted_target_quaternion = list(
                admitted_open_cell.get(
                    "authoritative_target_quaternion_world_xyzw"
                )
                or contact_quaternion
            )
            contact_quaternion = adopted_target_quaternion
            target_records = [{
                "phase_id": "contact_open",
                "target_position_world_m": adopted_target_position,
                "target_quaternion_world_xyzw": adopted_target_quaternion,
                "joint_positions_rad": adopted_open,
            }]
            bounded_candidate = admitted_open_cell.get(
                "bounded_orientation_candidate"
            )
            if isinstance(bounded_candidate, Mapping):
                target_records = _bounded_orientation_joint_targets(
                    control_plan=selected_control_plan,
                    admitted_cell=admitted_open_cell,
                )
                for row in selected_control_plan[
                    "scripted_positive_actions"
                ]:
                    if (
                        isinstance(row, dict)
                        and str(row.get("phase_id") or "")
                        == "contact_close"
                    ):
                        row.pop(
                            "physx_dls_preferred_posture_joint_positions_rad",
                            None,
                        )
            scripted_pose_joint_targets = [
                row
                for row in scripted_pose_joint_targets
                if str(row.get("phase_id") or "")
                not in {record["phase_id"] for record in target_records}
            ]
            scripted_pose_joint_targets.extend(target_records)
            selected_control_plan, held_vectors = (
                _with_held_solved_contact_vectors(
                    control_plan=selected_control_plan,
                    scripted_pose_joint_targets=scripted_pose_joint_targets,
                )
            )
            result["held_solved_contact_vectors"] = held_vectors
            effective_control_plan, branch_replay = (
                _with_contact_entry_branch_replay(
                    control_plan=selected_control_plan,
                    scripted_pose_joint_targets=scripted_pose_joint_targets,
                    task_spec=scene_plan["task_spec"],
                    actuator_feasible_step_rad=actuator_feasible_step_vector,
                )
            )
            result["contact_entry_branch_replay"] = branch_replay
            result["control_plan_digest"] = effective_control_plan[
                "plan_digest"
            ]
            contact_row = next(
                row
                for row in effective_control_plan["scripted_positive_actions"]
                if isinstance(row, Mapping)
                and str(row.get("phase_id") or "") == "contact_open"
            )
            seed_posture = adopted_open

        # C36 localized the defect to a kinematic constant: at the solved
        # contact posture, across a tenfold stiffness range and with joint
        # tracking at 0.007 rad, the measured fingertip sat +13.0 mm off in a
        # single axis.  The solver hits its own target; its model of where the
        # fingertip is disagrees with PhysX.  So solve for the posture whose
        # *measured* fingertip reaches the sealed target, by folding each
        # measured residual back into the solver's target.  The arrival gate is
        # untouched -- this stops handing it a posture the model had wrong.
        if anchor_admitted:
            calibration = {
                "schema_version": "native_task_arena_measured_posture_calibration.v1",
                "status": "skipped",
                "reason": "measured_approach_anchor_inside_arrival_gate",
                "iterations": [],
            }
        else:
            try:
                from blueprint_pipeline.native_task_arena_actuator_sweep import (
                    calibrate_posture_to_measured_target,
                )

                calibration = (
                    calibrate_posture_to_measured_target(
                        environment=episode_environment,
                        solve=_solve_contact,
                        target_position_world_m=contact_row[
                            "target_position_world_m"
                        ],
                        seed_joint_positions_rad=seed_posture,
                        gripper_open_command=float(gripper["open_command"]),
                        max_joint_delta_rad=float(
                            contact_row["max_joint_delta_rad"]
                        ),
                        max_joint_setpoint_lead_rad=float(
                            contact_row["max_joint_setpoint_lead_rad"]
                        ),
                        arrival_tolerance_m=contact_tolerance,
                    )
                    if seed_posture is not None
                    else {
                        "status": "unavailable",
                        "reason": "contact_posture_unsolved",
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - diagnostic only
                calibration = {
                    "schema_version": "native_task_arena_measured_posture_calibration.v1",
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "iterations": [],
                }
        result["contact_posture_measured_calibration"] = calibration

        _announce("contact_target_reachability_probe")
        if anchor_admitted:
            # C50 closed the controller question: the exact selected posture
            # was commanded, FK matched measured TCP to 1.34 micrometres, and
            # the arm still stopped 22 mm short under 543 N of two-pad contact.
            # Localize that collision frontier from the measured-clear anchor
            # instead of spending another run on one endpoint. Cells are
            # reset-isolated, stop at 50 N, and do not consume the scored task
            # action budget.
            frontier_offsets = _contact_frontier_offsets(anchor_offset or [])
            preposition_target = [
                float(contact_row["target_position_world_m"][axis])
                + float(anchor_offset[axis])
                for axis in range(3)
            ]
            try:
                reach_probe = probe_target_reachability(
                    environment=episode_environment,
                    solve=_solve_contact,
                    base_target_position_world_m=contact_row[
                        "target_position_world_m"
                    ],
                    seed_joint_positions_rad=seed_posture,
                    gripper_open_command=float(gripper["open_command"]),
                    max_joint_delta_rad=float(
                        contact_row["max_joint_delta_rad"]
                    ),
                    max_joint_setpoint_lead_rad=float(
                        contact_row["max_joint_setpoint_lead_rad"]
                    ),
                    offsets_m=frontier_offsets,
                    preposition_target_position_world_m=preposition_target,
                    abort_contact_force_n=50.0,
                    # The ordered cells walk from known-clear toward the
                    # authored endpoint. Once one cell makes task contact the
                    # boundary is bracketed; deeper cells only repeat a known
                    # collision and cannot improve the promoted safe standoff.
                    stop_after_first_contact_cell=True,
                )
                reach_probe["diagnostic_kind"] = (
                    "known_clear_anchor_to_authored_contact_force_frontier"
                )
            except BaseException as exc:  # noqa: BLE001 - diagnostic only
                reach_probe = {
                    "schema_version": (
                        "native_task_arena_target_reachability_probe.v1"
                    ),
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "cells": [],
                }
        else:
            try:
                reach_probe = (
                    probe_target_reachability(
                        environment=episode_environment,
                        solve=_solve_contact,
                        base_target_position_world_m=contact_row[
                            "target_position_world_m"
                        ],
                        seed_joint_positions_rad=seed_posture,
                        gripper_open_command=float(gripper["open_command"]),
                        max_joint_delta_rad=float(
                            contact_row["max_joint_delta_rad"]
                        ),
                        max_joint_setpoint_lead_rad=float(
                            contact_row["max_joint_setpoint_lead_rad"]
                        ),
                    )
                    if seed_posture is not None
                    else {
                        "schema_version": (
                            "native_task_arena_target_reachability_probe.v1"
                        ),
                        "status": "unavailable",
                        "reason": "contact_posture_unsolved",
                        "cells": [],
                    }
                )
            except BaseException as exc:  # noqa: BLE001 - diagnostic only
                reach_probe = {
                    "schema_version": (
                        "native_task_arena_target_reachability_probe.v1"
                    ),
                    "status": "unavailable",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "cells": [],
                }
        result["contact_target_reachability_probe"] = reach_probe

        # Use the last reset-isolated, zero-contact frontier cell: this is the
        # exact physical open pose the episode will promote, not merely the
        # off-sim branch that seeded the probe.
        close_sweep_preposition_cell = next(
            (
                cell
                for cell in reversed(reach_probe.get("cells") or [])
                if isinstance(cell, Mapping)
                and int(cell.get("contact_steps") or 0) == 0
                and isinstance(cell.get("joint_positions_rad"), list)
            ),
            None,
        )
        close_sweep_preposition = (
            close_sweep_preposition_cell.get("joint_positions_rad")
            if isinstance(close_sweep_preposition_cell, Mapping)
            else seed_posture
        )

        # C56 proved that off-sim chain continuity can select a mathematically
        # valid close branch whose tiny 0.012 rad tracking residual amplifies
        # into about 13 mm of TCP error.  Measure every solved close branch in
        # the same PhysX runtime, from the same qualified open posture, before
        # allowing one to become the held episode vector.
        contact_close_row = next(
            row
            for row in effective_control_plan["scripted_positive_actions"]
            if isinstance(row, Mapping)
            and str(row.get("phase_id") or "") == "contact_close"
        )
        close_contact_threshold = _contact_close_sweep_minimum_force_n(
            contact_close_row=contact_close_row,
            task_state_binding=scene_plan["task_state_binding"],
        )

        def _solve_closed_contact(target_position, seed_joints):
            return _solve_closed_contact_on_reference_branch(
                servo=servo,
                contact_close_row=contact_close_row,
                target_position_world_m=target_position,
                reference_joint_positions_rad=seed_joints,
            )

        close_postures = candidate_postures(
            controls_global_ik, phase_id="contact_close"
        )
        if not close_postures and seed_posture is not None:
            # The reset-isolated contact-open matrix has already proved this
            # posture is a physically safe starting branch.  Even when PINK
            # cannot solve the closed linkage, use that measured branch as the
            # null-space preference for the live closed-TCP controller.
            close_postures = [
                {
                    "posture_index": 0,
                    "seed_index": None,
                    "joint_positions_rad": list(seed_posture),
                    "posture_source": "physics_admitted_contact_open",
                }
            ]

        try:
            from blueprint_pipeline.native_task_arena_actuator_sweep import (
                candidate_postures,
                run_contact_close_posture_sweep,
            )

            close_posture_sweep = (
                run_contact_close_posture_sweep(
                    environment=episode_environment,
                    target_position_world_m=contact_close_row.get(
                        "arrival_target_position_world_m",
                        contact_close_row["target_position_world_m"],
                    ),
                    target_orientation_world_xyzw=contact_close_row[
                        "target_quaternion_world_xyzw"
                    ],
                    postures=close_postures,
                    preposition_joint_positions_rad=close_sweep_preposition,
                    # The sweep owns scalar validation and reports a typed
                    # input error.  Eager caller-side float conversions made
                    # C57/C58 lose the entire branch surface before the
                    # sweep's isolation code could run.
                    gripper_open_command=gripper["open_command"],
                    gripper_closed_command=gripper["closed_command"],
                    max_joint_delta_rad=contact_close_row[
                        "max_joint_delta_rad"
                    ],
                    max_joint_setpoint_lead_rad=contact_close_row[
                        "max_joint_setpoint_lead_rad"
                    ],
                    arrival_tolerance_m=contact_close_row[
                        "arrival_tolerance_m"
                    ],
                    orientation_tolerance_rad=(
                        contact_close_row.get(
                            "arrival_orientation_tolerance_rad"
                        )
                        or 0.08
                    ),
                    bilateral_contact_minimum_force_n=close_contact_threshold,
                    solve=_solve_closed_contact,
                    solver_target_position_world_m=contact_close_row[
                        "target_position_world_m"
                    ],
                    max_calibration_iterations=4,
                    bilateral_stability_steps=2,
                    # C73 showed the held solved-vector seam winning dispatch
                    # over the live PhysX TCP controller: every episode retry
                    # replayed joint targets even though the runtime reported
                    # contact_close as DLS-capable. Compare both controllers
                    # reset-isolated on every branch before adopting either.
                    compare_physx_dls=True,
                )
                if close_sweep_preposition is not None
                else {
                    "schema_version": (
                        "native_task_arena_contact_close_posture_sweep.v1"
                    ),
                    "status": "unavailable",
                    "reason": "contact_open_posture_unsolved",
                    "cells": [],
                }
            )
        except BaseException as exc:  # noqa: BLE001 - retain diagnostic gap
            close_posture_sweep = {
                "schema_version": (
                    "native_task_arena_contact_close_posture_sweep.v1"
                ),
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
                "cells": [],
            }
        close_posture_sweep["calibration_solver_selection_policy"] = (
            "minimum_joint_travel_from_measured_branch_subject_to_required_"
            "joint_limit_margin"
        )
        result["contact_close_posture_physics_sweep"] = close_posture_sweep
        close_sweep_best = close_posture_sweep.get("best_cell") or {}
        if (
            close_sweep_best.get("admitted") is True
            and isinstance(
                close_sweep_best.get("commanded_joint_positions_rad"), list
            )
        ):
            adopted_close = list(
                close_sweep_best["commanded_joint_positions_rad"]
            )
            if (
                close_sweep_best.get("controller_mode")
                == "live_physx_dls"
            ):
                selected_control_plan, dls_adoption = (
                    _with_live_physx_dls_contact_close(
                        control_plan=selected_control_plan,
                        preferred_posture_joint_positions_rad=adopted_close,
                    )
                )
                result["contact_close_controller_adoption"] = dls_adoption
                if dls_adoption.get("status") != "applied":
                    raise RuntimeError(
                        "native_task_controls_physx_dls_close_adoption_failed:"
                        + str(dls_adoption.get("reason"))
                    )
                prior_held = dict(result["held_solved_contact_vectors"])
                result["held_solved_contact_vectors"] = {
                    **prior_held,
                    "held_phase_ids": [
                        phase_id
                        for phase_id in prior_held.get("held_phase_ids") or []
                        if phase_id != "contact_close"
                    ],
                    "derived_control_plan_digest": selected_control_plan.get(
                        "plan_digest"
                    ),
                    "claim_boundary": (
                        "contact_open_commands_the_preflight_solved_posture;"
                        "contact_close_executes_the_reset_isolated_physics_"
                        "admitted_live_tcp_controller"
                    ),
                }
            else:
                scripted_pose_joint_targets = [
                    (
                        {**row, "joint_positions_rad": adopted_close}
                        if str(row.get("phase_id") or "") == "contact_close"
                        else row
                    )
                    for row in scripted_pose_joint_targets
                ]
                selected_control_plan, held_vectors = (
                    _with_held_solved_contact_vectors(
                        control_plan=selected_control_plan,
                        scripted_pose_joint_targets=scripted_pose_joint_targets,
                    )
                )
                result["held_solved_contact_vectors"] = held_vectors
                result["contact_close_controller_adoption"] = {
                    "schema_version": (
                        "native_task_controls_contact_close_controller_"
                        "adoption.v1"
                    ),
                    "status": "applied",
                    "controller_mode": "bounded_joint_replay",
                }
            effective_control_plan, branch_replay = (
                _with_contact_entry_branch_replay(
                    control_plan=selected_control_plan,
                    scripted_pose_joint_targets=scripted_pose_joint_targets,
                    task_spec=scene_plan["task_spec"],
                    actuator_feasible_step_rad=actuator_feasible_step_vector,
                )
            )
            result["contact_entry_branch_replay"] = branch_replay
            result["contact_close_posture_sweep_adopted"] = True
        else:
            result["contact_close_posture_sweep_adopted"] = False

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
            # C37 sealed the trap this closes: the episode executes the plan
            # compiled BEFORE calibration, so rewriting the posture list alone
            # adopted nothing -- contact entry replayed the model's branch and
            # missed by 70-114 mm while the calibrated posture measured 13 mm.
            # Adoption is only real when the plan is recompiled through the
            # same replay generator, and only claimed when that replay
            # actually applied.
            result["contact_entry_branch_replay_pre_calibration"] = result[
                "contact_entry_branch_replay"
            ]
            recompiled_plan, recompiled_replay = _with_contact_entry_branch_replay(
                control_plan=selected_control_plan,
                scripted_pose_joint_targets=scripted_pose_joint_targets,
                task_spec=scene_plan["task_spec"],
                actuator_feasible_step_rad=actuator_feasible_step_vector,
            )
            if recompiled_replay.get("status") == "applied":
                effective_control_plan = recompiled_plan
                result["contact_entry_branch_replay"] = recompiled_replay
                result["control_plan_digest"] = recompiled_plan["plan_digest"]
                result["contact_posture_calibration_adopted"] = True
            else:
                result["contact_posture_calibration_adopted"] = False
                result["contact_posture_calibration_adoption_blocker"] = str(
                    recompiled_replay.get("reason")
                )
        else:
            result["contact_posture_calibration_adopted"] = False

        # Start the paid episode from a point this same PhysX runtime already
        # measured inside the gate, then advance toward contact in small
        # increments.  This turns a final-pose miss into a measured success
        # frontier and avoids throwing away the reachability probe's answer.
        effective_control_plan, measured_frontier = (
            _with_measured_contact_frontier(
                control_plan=effective_control_plan,
                reachability_probe=reach_probe,
                task_spec=scene_plan["task_spec"],
                task_contact_minimum_force_n=close_contact_threshold,
                reclaimed_contact_steps=int(
                    result["contact_entry_branch_replay"].get(
                        "contact_phase_steps_reclaimed"
                    )
                    or 0
                ),
            )
        )
        result["measured_contact_frontier"] = measured_frontier
        if (
            measured_frontier.get("status") == "applied"
            and int(measured_frontier.get("replaced_branch_replay_rows") or 0) > 0
        ):
            result["contact_entry_branch_replay_pre_measured_anchor"] = result[
                "contact_entry_branch_replay"
            ]
            result["contact_entry_branch_replay"] = {
                "schema_version": (
                    "native_task_controls_contact_entry_branch_replay.v1"
                ),
                "status": "replaced_by_measured_anchor",
                "replaced_rows": int(
                    measured_frontier["replaced_branch_replay_rows"]
                ),
                "replacement_phase_id": MEASURED_CONTACT_ENTRY_PHASE_ID,
                "replacement_control_plan_digest": effective_control_plan[
                    "plan_digest"
                ],
            }
        result["control_plan_digest"] = effective_control_plan["plan_digest"]

        # The earlier acquisition grid displaced the command target and then
        # required both that displaced target and the authored target to sit
        # inside 5 mm. Any cell displaced by more than 10 mm is mathematically
        # unable to pass both gates, and the interrupted symmetric sweep
        # already exhausted the geometrically admissible centre/5/10 mm
        # slices. The measured closed-contact calibration above now varies the
        # thing that actually failed -- the commanded joint posture under the
        # closed linkage -- while the authored target stays fixed. Do not pay
        # again for a target-offset surface whose admissible region is closed.
        _announce("contact_acquisition_sweep")
        authored_close_target = [
            float(value)
            for value in contact_close_row.get(
                "arrival_target_position_world_m",
                contact_close_row["target_position_world_m"],
            )
        ]
        open_target = [
            float(value) for value in contact_row["target_position_world_m"]
        ]
        pad_centers = (
            close_sweep_preposition_cell.get(
                "measured_gripper_pad_centers_world_m"
            )
            if isinstance(close_sweep_preposition_cell, Mapping)
            else None
        )
        approach_axis, jaw_axis, lateral_axis = _contact_acquisition_axes(
            control_plan=effective_control_plan,
            authored_open_target=open_target,
            authored_close_target=authored_close_target,
            pad_centers=pad_centers,
        )
        try:
            from blueprint_pipeline.native_task_arena_actuator_sweep import (
                run_contact_acquisition_sweep,
            )

            acquisition_progress_path = (
                output_root / "contact_acquisition_sweep.progress.v1.json"
            )

            def _contact_acquisition_progress(
                progress: Mapping[str, Any],
            ) -> None:
                _persist_progress(acquisition_progress_path, progress)
                _announce_contact_acquisition_cell(progress)

            contact_acquisition = (
                {
                    "schema_version": (
                        "native_task_arena_contact_acquisition_sweep.v1"
                    ),
                    "status": "skipped",
                    "reason": (
                        "superseded_by_closed_contact_measured_calibration"
                    ),
                    "cells": [],
                    "provider_mutation_performed": False,
                    "claim_boundary": (
                        "the_target_offset_grid_cannot_admit_cells_whose_"
                        "candidate_and_authored_5mm_balls_do_not_intersect;"
                        "closed_contact_calibration_keeps_the_authored_target_"
                        "fixed_and_measures_the_failed_joint_posture_instead"
                    ),
                }
                if close_posture_sweep.get("calibration_enabled") is True
                else
                run_contact_acquisition_sweep(
                    environment=episode_environment,
                    authored_target_position_world_m=authored_close_target,
                    command_target_position_world_m=contact_close_row[
                        "target_position_world_m"
                    ],
                    target_orientation_world_xyzw=contact_close_row[
                        "target_quaternion_world_xyzw"
                    ],
                    preposition_joint_positions_rad=close_sweep_preposition,
                    approach_axis_world=approach_axis,
                    jaw_axis_world=jaw_axis,
                    lateral_axis_world=lateral_axis,
                    gripper_open_command=gripper["open_command"],
                    gripper_closed_command=gripper["closed_command"],
                    max_joint_delta_rad=contact_close_row[
                        "max_joint_delta_rad"
                    ],
                    max_joint_setpoint_lead_rad=contact_close_row[
                        "max_joint_setpoint_lead_rad"
                    ],
                    arrival_tolerance_m=contact_close_row[
                        "arrival_tolerance_m"
                    ],
                    orientation_tolerance_rad=(
                        contact_close_row.get(
                            "arrival_orientation_tolerance_rad"
                        )
                        or 0.08
                    ),
                    bilateral_contact_minimum_force_n=(
                        close_contact_threshold
                    ),
                    stop_after_admitted_cells=1,
                    progress_callback=_contact_acquisition_progress,
                )
                if (
                    close_sweep_preposition is not None
                    and approach_axis is not None
                    and jaw_axis is not None
                    and lateral_axis is not None
                )
                else {
                    "schema_version": (
                        "native_task_arena_contact_acquisition_sweep.v1"
                    ),
                    "status": "unavailable",
                    "reason": "contact_acquisition_basis_unresolved",
                    "cells": [],
                }
            )
        except BaseException as exc:  # noqa: BLE001 - diagnostic only
            contact_acquisition = {
                "schema_version": (
                    "native_task_arena_contact_acquisition_sweep.v1"
                ),
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
                "cells": [],
            }
        result["contact_acquisition_sweep"] = contact_acquisition
        effective_control_plan, contact_acquisition_adoption = (
            _with_contact_acquisition_candidate(
                control_plan=effective_control_plan,
                sweep=contact_acquisition,
            )
        )
        result["contact_acquisition_adoption"] = (
            contact_acquisition_adoption
        )
        result["control_plan_digest"] = effective_control_plan["plan_digest"]
        _announce(
            "contact_acquisition_sweep",
            (
                "completed"
                if contact_acquisition.get("status") == "measured"
                else "blocked"
            ),
        )
        _announce(
            "contact_posture_actuator_sweep",
            (
                "completed"
                if sweep.get("status") in {"measured", "skipped"}
                else "blocked"
            ),
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
