"""Freeze the native articulation readback probe before any simulator spend.

The statically admitted articulated candidate still has to prove itself in a
native PhysX/Isaac run: the articulation root and joint graph must read back as
authored, the commanded task joint must reach the required opening while the
locked joints hold, contacts must be stable with no initial penetration, and a
reset must replay to a deterministic final state.

This module writes that probe as frozen, digest-bound inputs: a blank-stage
diagnostic that proves the runtime can bring up a physics scene at all, an
articulation stage that references the exact candidate bytes, and a spec
listing every required readback with its preregistered expectation. It never
executes anything and never asserts a result; a spec whose expectations
contradict the candidate's authored limits fails closed here rather than after
paid time.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .decision_evidence_contracts import canonical_digest


ARTICULATED_NATIVE_PROBE_SCHEMA_VERSION = "articulated_native_probe_spec.v1"
REQUIRED_READBACKS = (
    "articulation_root_identity",
    "joint_count_and_types",
    "task_joint_identity",
    "locked_joint_identity",
    "joint_axis_and_limits",
    "locked_joint_motion_within_tolerance",
    "commanded_sweep_reaches_maximum",
    "contact_stability",
    "no_initial_penetration",
    "reset_replay_within_tolerance",
    "deterministic_final_state",
)
_BLANK_STAGE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def PhysicsScene "physics_scene"
    {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }
}
"""


class ArticulatedNativeProbeError(ValueError):
    """Stable, sorted native-probe construction failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _articulation_stage(
    candidate_relative_path: str,
    default_prim_name: str,
    probe_drive: Mapping[str, Any] | None,
) -> str:
    """Overlay a physics scene, and probe-time actuation the asset must not own.

    The shipped asset gives its task joint friction, not a position servo, so
    that a task scored on the door staying open after release measures the
    robot rather than a stiffness constant. The probe still has to command that
    joint, so the actuation it needs lives here - in the probe's own overlay,
    digest-bound with the rest of the frozen spec and absent from the asset.
    """

    actuation = ""
    if probe_drive:
        joint_path = str(probe_drive["joint_prim_path"])
        relative = joint_path.split(f"/{default_prim_name}/", 1)[-1]
        segments = relative.split("/")
        opening = "".join(
            f'{"    " * (index + 1)}over "{segment}"\n{"    " * (index + 1)}{{\n'
            for index, segment in enumerate(segments)
        )
        closing = "".join(
            f'{"    " * (index + 1)}}}\n' for index in reversed(range(len(segments)))
        )
        depth = "    " * (len(segments) + 1)
        body = (
            f'{depth}uniform token[] apiSchemas = ["PhysicsDriveAPI:angular"]\n'
            f'{depth}uniform token drive:angular:physics:type = "force"\n'
            f"{depth}float drive:angular:physics:stiffness = "
            f"{float(probe_drive['stiffness'])}\n"
            f"{depth}float drive:angular:physics:damping = "
            f"{float(probe_drive['damping'])}\n"
            f"{depth}float drive:angular:physics:maxForce = "
            f"{float(probe_drive['max_force'])}\n"
        )
        actuation = "\n" + opening + body + closing
    return f"""#usda 1.0
(
    subLayers = [@{candidate_relative_path}@]
    defaultPrim = "{default_prim_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

over "{default_prim_name}"
{{
    def PhysicsScene "physics_scene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
{actuation}}}
"""


def materialize_articulated_native_probe(
    *,
    candidate_usd_path: str | Path,
    destination: str | Path,
    task_joint_prim_path: str,
    locked_joint_prim_paths: Sequence[str],
    commanded_sweep_degrees: Sequence[float],
    reset_joint_positions_rad: Mapping[str, float],
    locked_joint_motion_tolerance_rad: float,
    settle_samples: int,
    control_frequency_hz: float,
    probe_drive_stiffness: float = 0.0,
    probe_drive_damping: float = 0.0,
    probe_drive_max_force: float = 0.0,
    fixed_step_seconds: float = 1.0 / 120.0,
) -> dict[str, Any]:
    """Write the frozen native probe stages and spec for one articulated asset."""

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedNativeProbeError(
            ["articulated_native_probe_openusd_runtime_missing"]
        ) from exc

    candidate = Path(candidate_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise ArticulatedNativeProbeError(["articulated_native_probe_candidate_missing"])

    errors: list[str] = []
    stage = Usd.Stage.Open(str(candidate))
    if stage is None:
        raise ArticulatedNativeProbeError(
            ["articulated_native_probe_candidate_unreadable"]
        )
    roots = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    joints = {
        str(prim.GetPath()): prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    }
    if len(roots) != 1:
        errors.append(
            f"articulated_native_probe_articulation_root_count_invalid:{len(roots)}"
        )
    task_joint = joints.get(str(task_joint_prim_path))
    if task_joint is None:
        errors.append(
            f"articulated_native_probe_task_joint_not_found:{task_joint_prim_path}"
        )
    locked = [str(path) for path in locked_joint_prim_paths]
    for path in locked:
        if path not in joints:
            errors.append(f"articulated_native_probe_locked_joint_not_found:{path}")
    if set(locked) | {str(task_joint_prim_path)} != set(joints):
        errors.append("articulated_native_probe_joint_partition_incomplete")

    sweep: list[float] = []
    for value in commanded_sweep_degrees:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append("articulated_native_probe_sweep_invalid")
            sweep = []
            break
        sweep.append(float(value))
    if sweep and (sweep[0] != 0.0 or any(b <= a for a, b in zip(sweep, sweep[1:]))):
        errors.append("articulated_native_probe_sweep_invalid")

    limits_deg: list[float] = []
    axis_token = ""
    if task_joint is not None and task_joint.IsA(UsdPhysics.RevoluteJoint):
        revolute = UsdPhysics.RevoluteJoint(task_joint)
        lower = revolute.GetLowerLimitAttr().Get()
        upper = revolute.GetUpperLimitAttr().Get()
        axis_attribute = task_joint.GetAttribute("physics:axis")
        axis_token = (
            str(axis_attribute.Get())
            if axis_attribute and axis_attribute.HasAuthoredValue()
            else ""
        )
        if lower is None or upper is None or not axis_token:
            errors.append("articulated_native_probe_task_joint_limits_missing")
        else:
            limits_deg = [float(lower), float(upper)]
            if sweep and (sweep[0] < limits_deg[0] or sweep[-1] > limits_deg[1]):
                errors.append(
                    "articulated_native_probe_commanded_sweep_outside_joint_limits:"
                    f"{sweep[-1]!r}>{limits_deg[1]!r}"
                )
    elif task_joint is not None:
        errors.append("articulated_native_probe_task_joint_not_revolute")

    resets: dict[str, float] = {}
    for path, value in reset_joint_positions_rad.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"articulated_native_probe_reset_invalid:{path}")
            continue
        resets[str(path)] = float(value)
        joint_prim = joints.get(str(path))
        if joint_prim is None or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        revolute = UsdPhysics.RevoluteJoint(joint_prim)
        lower = revolute.GetLowerLimitAttr().Get()
        upper = revolute.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            continue
        degrees = math.degrees(float(value))
        if degrees < float(lower) - 1e-9 or degrees > float(upper) + 1e-9:
            errors.append(
                f"articulated_native_probe_reset_position_outside_joint_limits:{path}"
            )
    if set(resets) != set(joints):
        errors.append("articulated_native_probe_reset_does_not_cover_every_joint")

    tolerance = float(locked_joint_motion_tolerance_rad)
    frequency = float(control_frequency_hz)
    step = float(fixed_step_seconds)
    if (
        not isinstance(settle_samples, int)
        or settle_samples < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
        or not math.isfinite(frequency)
        or frequency <= 0.0
        or not math.isfinite(step)
        or step <= 0.0
    ):
        errors.append("articulated_native_probe_settle_parameters_invalid")
    if errors:
        raise ArticulatedNativeProbeError(errors)

    ensure_dir(output)
    candidate_copy = output / candidate.name
    candidate_copy.write_bytes(candidate.read_bytes())
    blank_path = output / "blank_physics_stage.usda"
    blank_path.write_text(_BLANK_STAGE, encoding="utf-8")
    articulation_path = output / "articulation_stage.usda"
    probe_drive = (
        {
            "joint_prim_path": str(task_joint_prim_path),
            "stiffness": float(probe_drive_stiffness),
            "damping": float(probe_drive_damping),
            "max_force": float(probe_drive_max_force),
        }
        if float(probe_drive_stiffness) > 0.0 or float(probe_drive_damping) > 0.0
        else None
    )
    articulation_path.write_text(
        _articulation_stage(
            candidate.name, stage.GetDefaultPrim().GetName(), probe_drive
        ),
        encoding="utf-8",
    )

    joint_types = {
        path: (
            "revolute"
            if prim.IsA(UsdPhysics.RevoluteJoint)
            else "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "other"
        )
        for path, prim in sorted(joints.items())
    }
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()

    receipt: dict[str, Any] = {
        "schema_version": ARTICULATED_NATIVE_PROBE_SCHEMA_VERSION,
        "status": "frozen_not_executed",
        "candidate_usd_path": str(candidate),
        "candidate_usd_sha256": _sha256(candidate),
        "spec_path": str(output / "articulated_native_probe_spec.json"),
        "stages": {
            "blank_stage": {
                "path": str(blank_path),
                "sha256": _sha256(blank_path),
                "purpose": "prove_the_runtime_can_bring_up_a_physics_scene_before_the_asset",
            },
            "articulation_stage": {
                "path": str(articulation_path),
                "sha256": _sha256(articulation_path),
                "purpose": "reference_the_exact_candidate_and_add_only_a_physics_scene",
            },
            "candidate_copy": {
                "path": str(candidate_copy),
                "sha256": _sha256(candidate_copy),
                "purpose": "immutable_probe_input_copy",
            },
        },
        "expected": {
            "articulation_root_prim_path": roots[0],
            "assembly_joint_count": len(joints),
            "joint_types": joint_types,
            "task_joint_prim_path": str(task_joint_prim_path),
            "locked_joint_prim_paths": sorted(locked),
            "task_joint_axis": axis_token,
            "task_joint_limits_deg": limits_deg,
            "commanded_sweep_degrees": sweep,
            "maximum_commanded_degrees": sweep[-1] if sweep else None,
            "reset_joint_positions_rad": dict(sorted(resets.items())),
            "locked_joint_motion_tolerance_rad": tolerance,
            "candidate_world_bound_min_m": [
                float(bound.GetMin()[axis]) for axis in range(3)
            ],
            "candidate_world_bound_max_m": [
                float(bound.GetMax()[axis]) for axis in range(3)
            ],
        },
        "settle": {
            "samples": settle_samples,
            "control_frequency_hz": frequency,
            "window_seconds": settle_samples / frequency,
            "fixed_step_seconds": step,
        },
        "probe_drive": probe_drive,
        "required_readbacks": list(REQUIRED_READBACKS),
        "claim_boundary": {
            "frozen_before_execution": True,
            "native_simulator_qualified": False,
            "physical_equivalence_proven": False,
            "spec_is_not_a_result": True,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["spec_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ARTICULATED_NATIVE_PROBE_SCHEMA_VERSION",
    "ArticulatedNativeProbeError",
    "REQUIRED_READBACKS",
    "materialize_articulated_native_probe",
]
