"""Author the joint drives an articulated replacement needs to behave.

Isaac showed that the 840796 twin's joints had no drive at all: a position
target was accepted and ignored, and the door sat at zero through the whole
commanded sweep while every other readback passed. NVIDIA's Joint Agent
authors topology and explicitly not drives, so nothing upstream supplies one.

What to author is not obvious, and getting it wrong would quietly corrupt the
evaluation. The frozen task scores the door *staying open after release*, so a
position servo on the commanded hinge would hold it there no matter what the
robot did - the run would be measuring our stiffness constant. The commanded
joint therefore gets damping only: viscous hinge friction that resists motion
and leaves the door where it is put, exactly like the appliance it replaces.
Joints the task locks are a different case and may be held at their reset
position, because holding them is the intent.

That leaves the probe unable to command the joint by position, which is
correct: probe-time actuation belongs in the probe's own overlay, not baked
into the shipped asset.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest


JOINT_DRIVE_SCHEMA_VERSION = "articulated_joint_drives.v1"
ROLE_TASK_JOINT = "task_joint_free_with_friction"
ROLE_LOCKED_JOINT = "locked_joint_held_closed"
ROLES = (ROLE_TASK_JOINT, ROLE_LOCKED_JOINT)


class ArticulatedJointDriveError(ValueError):
    """Stable, sorted joint-drive authoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _number(value: Any, default: float, error: str) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArticulatedJointDriveError([error])
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ArticulatedJointDriveError([error])
    return number


def author_articulated_joint_drives(
    *,
    source_usd_path: str | Path,
    destination: str | Path,
    drives: Sequence[dict[str, Any]],
    drive_name: str = "angular",
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply per-joint drives on a copy, refusing to servo a commanded joint."""

    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedJointDriveError(
            ["articulated_joint_drive_openusd_runtime_missing"]
        ) from exc

    source = Path(source_usd_path).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ArticulatedJointDriveError(["articulated_joint_drive_source_missing"])
    if output == source:
        raise ArticulatedJointDriveError(
            ["articulated_joint_drive_destination_is_source"]
        )
    if not drives:
        raise ArticulatedJointDriveError(["articulated_joint_drive_specs_missing"])

    errors: list[str] = []
    specs: list[dict[str, Any]] = []
    for index, spec in enumerate(drives):
        role = str(spec.get("role") or "")
        path = str(spec.get("joint_prim_path") or "")
        if role not in ROLES:
            errors.append(f"articulated_joint_drive_role_unsupported:{role}")
            continue
        if not path:
            errors.append(f"articulated_joint_drive_{index}_joint_path_missing")
            continue
        stiffness = _number(
            spec.get("stiffness"), 0.0, "articulated_joint_drive_stiffness_invalid"
        )
        damping = _number(
            spec.get("damping"), 0.0, "articulated_joint_drive_damping_invalid"
        )
        if role == ROLE_TASK_JOINT and stiffness > 0.0:
            errors.append(
                "articulated_joint_drive_task_joint_must_not_be_position_servoed:"
                f"{path}"
            )
            continue
        if stiffness <= 0.0 and damping <= 0.0:
            errors.append(f"articulated_joint_drive_has_no_effect:{path}")
            continue
        specs.append(
            {
                "joint_prim_path": path,
                "role": role,
                "stiffness": stiffness,
                "damping": damping,
                "target_position_degrees": _number(
                    spec.get("target_position_degrees"),
                    0.0,
                    "articulated_joint_drive_target_invalid",
                ),
                "max_force": _number(
                    spec.get("max_force"),
                    0.0,
                    "articulated_joint_drive_max_force_invalid",
                ),
                "position_servo_enabled": role == ROLE_LOCKED_JOINT,
                "holds_position_without_actuation": role == ROLE_LOCKED_JOINT,
                "resists_velocity_without_position_target": role == ROLE_TASK_JOINT,
            }
        )
    if errors:
        raise ArticulatedJointDriveError(errors)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise ArticulatedJointDriveError(["articulated_joint_drive_source_unreadable"])

    for spec in specs:
        prim = stage.GetPrimAtPath(spec["joint_prim_path"])
        if not prim.IsValid() or not prim.IsA(UsdPhysics.Joint):
            errors.append(
                f"articulated_joint_drive_joint_not_found:{spec['joint_prim_path']}"
            )
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(spec["stiffness"]))
        drive.CreateDampingAttr().Set(float(spec["damping"]))
        drive.CreateTargetPositionAttr().Set(float(spec["target_position_degrees"]))
        if spec["max_force"] > 0.0:
            drive.CreateMaxForceAttr().Set(float(spec["max_force"]))
    if errors:
        output.unlink(missing_ok=True)
        raise ArticulatedJointDriveError(errors)

    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(output))

    receipt: dict[str, Any] = {
        "schema_version": JOINT_DRIVE_SCHEMA_VERSION,
        "status": "articulated_joint_drives_authored",
        "source_usd_path": str(source),
        "source_usd_sha256": _sha256(source),
        "driven_usd_path": str(output),
        "driven_usd_sha256": _sha256(output),
        "drive_name": drive_name,
        "drives": specs,
        "preserved": {
            "assembly_joint_count": len(
                [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
            ),
            "articulation_root_count": len(
                [
                    p
                    for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI)
                ]
            ),
        },
        "claim_boundary": {
            "drive_constants_are_authored_not_measured": True,
            "task_joint_is_not_position_servoed": True,
            "probe_time_actuation_belongs_in_the_probe_overlay": True,
            "native_simulator_qualified": False,
        },
        "receipt_path": str(
            Path(receipt_path).expanduser().resolve()
            if receipt_path is not None
            else output.with_name(output.stem + "_joint_drive_receipt.json")
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(Path(receipt["receipt_path"]), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "ArticulatedJointDriveError",
    "JOINT_DRIVE_SCHEMA_VERSION",
    "ROLES",
    "ROLE_LOCKED_JOINT",
    "ROLE_TASK_JOINT",
    "author_articulated_joint_drives",
]
