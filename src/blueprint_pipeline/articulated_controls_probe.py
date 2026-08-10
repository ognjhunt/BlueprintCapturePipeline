"""Freeze the pair of controls that decide whether a door task is runnable.

The question a control pair answers is not "can something open this door" but
"does opening it require doing something". Those come apart in both directions,
and each failure looks like a result rather than a bug: a door with a mis-signed
drive falls open on its own and every policy scores a perfect success, while a
door welded shut by a stray collider scores every policy at zero. Running the
negative and the positive from one frozen spec is what tells them apart.

Two properties of the positive matter more than they look.

The force has to stop before the outcome is judged. The task is that the door
*stays* where it is put, so a probe that keeps pushing through the measurement
window is scoring its own pusher. Release, settle, then read.

And the force has to actually beat the gasket - but only briefly. Sizing one
constant force for the whole motion is impossible here, and the arithmetic says
why: the 24 N needed to crack this seal leaves the door coasting 85 degrees
past release, well beyond both the success window and the authored limit, while
any force gentle enough to stop in the window cannot break the seal at all.

That is not a modelling artifact, it is what the measured force trace looks
like - a sharp peak in the first few degrees and almost nothing after. So the
positive is a schedule: yank to break the seal, ease off, release early enough
that the coast lands inside the window. A schedule whose first phase cannot
break the gasket is refused here rather than burning a launch.

What this deliberately does not do is grasp. Force is applied at the handle
point directly, which isolates the door's own dynamics - drives, seal, and
whether it holds after release - from whether a gripper can hold on. Those are
separate questions and the second one is worth failing separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest


CONTROLS_PROBE_SCHEMA_VERSION = "articulated_controls_probe_spec.v1"
CONTROLS_PROBE_SPEC_FILENAME = "articulated_controls_probe_spec.json"
ZERO_ACTION_NEGATIVE = "zero_action_negative"
FORCED_POSITIVE = "forced_positive"
REQUIRED_CONTROLS_READBACKS = (
    "articulation_root_identity",
    "task_joint_identity",
    "zero_action_door_stays_shut",
    "forced_positive_reaches_success_window",
    "forced_positive_holds_after_release",
    "seal_resists_before_breakaway",
    "no_initial_penetration",
    "deterministic_replay_within_tolerance",
)
DEFAULT_SETTLE_STEPS = 120
DEFAULT_DRIVE_STEPS = 480
DEFAULT_PHYSICS_DT_S = 1.0 / 120.0


class ArticulatedControlsProbeError(ValueError):
    """Stable, sorted controls-probe authoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _vector(value: Any, error: str) -> list[float]:
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedControlsProbeError([error]) from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ArticulatedControlsProbeError([error])
    return values


def build_articulated_controls_probe(
    *,
    twin_usd_path: str | Path,
    destination: str | Path,
    task_joint_prim_path: str,
    task_link_prim_path: str,
    handle_grasp_point_local_m: Sequence[float],
    hinge_point_local_m: Sequence[float],
    hinge_axis_local: Sequence[float],
    target_open_angle_degrees: float,
    success_angle_window_degrees: Sequence[float],
    positive_force_schedule: Sequence[dict[str, Any]],
    seal_breakaway_torque_n_m: float = 0.0,
    seal_angular_width_degrees: float = 0.0,
    drive_steps: int = DEFAULT_DRIVE_STEPS,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    physics_dt_s: float = DEFAULT_PHYSICS_DT_S,
) -> dict[str, Any]:
    """Stage the twin and freeze a negative/positive pair against it."""

    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedControlsProbeError(
            ["articulated_controls_probe_openusd_runtime_missing"]
        ) from exc

    source = Path(twin_usd_path).expanduser().resolve()
    root = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ArticulatedControlsProbeError(["articulated_controls_probe_twin_missing"])

    handle = _vector(
        handle_grasp_point_local_m, "articulated_controls_probe_handle_invalid"
    )
    hinge = _vector(hinge_point_local_m, "articulated_controls_probe_hinge_invalid")
    axis = _vector(hinge_axis_local, "articulated_controls_probe_axis_invalid")
    axis_length = math.sqrt(sum(value * value for value in axis))
    if axis_length <= 0.0:
        raise ArticulatedControlsProbeError(["articulated_controls_probe_axis_invalid"])
    axis = [value / axis_length for value in axis]

    errors: list[str] = []
    try:
        window = [float(value) for value in success_angle_window_degrees]
    except (TypeError, ValueError):
        window = []
    if len(window) != 2 or not all(math.isfinite(v) for v in window) or window[0] >= window[1]:
        errors.append("articulated_controls_probe_success_window_invalid")
        window = [0.0, 0.0]
    seal_torque = float(seal_breakaway_torque_n_m or 0.0)
    seal_width = float(seal_angular_width_degrees or 0.0)

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ArticulatedControlsProbeError(
            ["articulated_controls_probe_twin_unreadable"]
        )
    # The root prim's name belongs to the asset. Writing a fixed one here would
    # silently produce an overlay that composes over nothing on any twin whose
    # generator picked a different name.
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ArticulatedControlsProbeError(
            ["articulated_controls_probe_twin_has_no_default_prim"]
        )
    default_prim_name = default_prim.GetName()
    joint_prim = stage.GetPrimAtPath(str(task_joint_prim_path))
    if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.Joint):
        errors.append(
            f"articulated_controls_probe_task_joint_missing:{task_joint_prim_path}"
        )
        authored_limit = 0.0
    else:
        authored_limit = float(
            UsdPhysics.RevoluteJoint(joint_prim).GetUpperLimitAttr().Get() or 0.0
        )
    if not stage.GetPrimAtPath(str(task_link_prim_path)).IsValid():
        errors.append(
            f"articulated_controls_probe_task_link_missing:{task_link_prim_path}"
        )

    if authored_limit and window[1] > authored_limit:
        # Judging success past the limit measures the solver's constraint
        # handling, not the door.
        errors.append(
            "articulated_controls_probe_success_window_beyond_authored_limit:"
            f"{window[1]}>{authored_limit}"
        )

    offset = [handle[index] - hinge[index] for index in range(3)]
    axial = sum(offset[index] * axis[index] for index in range(3))
    radial = [offset[index] - axis[index] * axial for index in range(3)]
    lever_arm = math.sqrt(sum(value * value for value in radial))
    if lever_arm <= 1e-6:
        errors.append("articulated_controls_probe_handle_on_hinge_axis")
        lever_arm = 1.0
    phases: list[dict[str, Any]] = []
    for index, raw in enumerate(positive_force_schedule or []):
        try:
            phase_force = float(raw["handle_force_n"])
            until = float(raw["until_angle_degrees"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"articulated_controls_probe_phase_invalid:{index}")
            continue
        if not math.isfinite(phase_force) or not math.isfinite(until):
            errors.append(f"articulated_controls_probe_phase_invalid:{index}")
            continue
        phases.append(
            {
                "phase_index": index,
                "handle_force_n": phase_force,
                "hinge_torque_n_m": phase_force * lever_arm,
                "until_angle_degrees": until,
            }
        )
    if not phases:
        errors.append("articulated_controls_probe_positive_schedule_missing")
    else:
        if [row["until_angle_degrees"] for row in phases] != sorted(
            row["until_angle_degrees"] for row in phases
        ):
            errors.append("articulated_controls_probe_phase_angles_not_increasing")
        opening_torque = phases[0]["hinge_torque_n_m"]
        if seal_torque > 0.0 and opening_torque <= seal_torque:
            errors.append(
                "articulated_controls_probe_force_below_seal_breakaway:"
                f"{opening_torque:.3f}<={seal_torque:.3f}"
            )
        if phases[-1]["until_angle_degrees"] > window[1]:
            errors.append(
                "articulated_controls_probe_release_after_success_window:"
                f"{phases[-1]['until_angle_degrees']}>{window[1]}"
            )
    if errors:
        raise ArticulatedControlsProbeError(errors)

    ensure_dir(root)
    twin_copy = root / source.name
    shutil.copy2(source, twin_copy)
    controls_stage = root / "controls_stage.usda"
    controls_stage.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@{source.name}@]\n"
        f'    defaultPrim = "{default_prim_name}"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        ")\n"
        "\n"
        f'over "{default_prim_name}"\n'
        "{\n"
        '    def PhysicsScene "physics_scene"\n'
        "    {\n"
        "        vector3f physics:gravityDirection = (0, 0, -1)\n"
        "        float physics:gravityMagnitude = 9.81\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    blank_stage = root / "blank_physics_stage.usda"
    blank_stage.write_text(
        "#usda 1.0\n"
        "(\n"
        '    defaultPrim = "World"\n'
        "    metersPerUnit = 1\n"
        '    upAxis = "Z"\n'
        ")\n"
        "\n"
        'def Xform "World"\n'
        "{\n"
        '    def PhysicsScene "physics_scene"\n'
        "    {\n"
        "        vector3f physics:gravityDirection = (0, 0, -1)\n"
        "        float physics:gravityMagnitude = 9.81\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    receipt: dict[str, Any] = {
        "schema_version": CONTROLS_PROBE_SCHEMA_VERSION,
        "status": "frozen_not_executed",
        "stages": {
            "blank_stage": {
                "path": str(blank_stage),
                "sha256": _sha256(blank_stage),
            },
            "twin_copy": {"path": str(twin_copy), "sha256": _sha256(twin_copy)},
            "controls_stage": {
                "path": str(controls_stage),
                "sha256": _sha256(controls_stage),
            },
        },
        "expected": {
            "task_joint_prim_path": str(task_joint_prim_path),
            "task_link_prim_path": str(task_link_prim_path),
            "authored_upper_limit_degrees": authored_limit,
        },
        "geometry": {
            "hinge_point_local_m": hinge,
            "hinge_axis_local_unit": axis,
            "handle_grasp_point_local_m": handle,
            "handle_force_direction_local": [
                axis[1] * radial[2] - axis[2] * radial[1],
                axis[2] * radial[0] - axis[0] * radial[2],
                axis[0] * radial[1] - axis[1] * radial[0],
            ],
            "lever_arm_m": lever_arm,
        },
        "seal": {
            "breakaway_torque_n_m": seal_torque,
            "angular_width_degrees": seal_width,
            "applied_by": "runtime_per_step_external_torque",
        },
        "controls": {
            ZERO_ACTION_NEGATIVE: {
                "control_id": ZERO_ACTION_NEGATIVE,
                "applied_handle_force_n": 0.0,
                "force_schedule": [],
                "drive_steps": int(drive_steps),
                "settle_steps": int(settle_steps),
                "release_before_settle": True,
                "expected_outcome": "door_does_not_reach_success_window",
            },
            FORCED_POSITIVE: {
                "control_id": FORCED_POSITIVE,
                "force_schedule": phases,
                "applied_handle_force_n": phases[0]["handle_force_n"],
                "applied_hinge_torque_n_m": phases[0]["hinge_torque_n_m"],
                "drive_steps": int(drive_steps),
                "settle_steps": int(settle_steps),
                # Force stops before the window is read, or the probe scores
                # its own pusher rather than whether the door stays put.
                "release_before_settle": True,
                "target_open_angle_degrees": float(target_open_angle_degrees),
                "expected_outcome": "door_holds_inside_success_window",
            },
        },
        "success_angle_window_degrees": window,
        "physics_dt_s": float(physics_dt_s),
        "required_readbacks": list(REQUIRED_CONTROLS_READBACKS),
        "claim_boundary": {
            "force_is_applied_at_the_handle_not_grasped": True,
            "grasp_feasibility_is_a_separate_question": True,
            "door_dynamics_isolated_from_gripper_capability": True,
            "native_simulator_executed": False,
        },
        "spec_digest": "",
    }
    receipt["spec_digest"] = canonical_digest(receipt, digest_field="spec_digest")
    write_json(root / CONTROLS_PROBE_SPEC_FILENAME, receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedControlsProbeError",
    "CONTROLS_PROBE_SCHEMA_VERSION",
    "CONTROLS_PROBE_SPEC_FILENAME",
    "FORCED_POSITIVE",
    "REQUIRED_CONTROLS_READBACKS",
    "ZERO_ACTION_NEGATIVE",
    "build_articulated_controls_probe",
]
