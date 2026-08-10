"""Refuse placed assets whose placement the spawner would silently erase.

Isaac Lab's spawn path authors a local translate and orient on the prim it
creates - the values come from the asset's initial pose, which for a placed
asset is deliberately zero. USD composition ranks that local opinion above
anything the referenced file says about the same attribute. So an asset that
carries its placement as a transform on its default prim opens at the right
pose in usdview and spawns at the world origin in the simulator.

rt51 and rt52 each paid GPU money to observe that difference. The probe below
observes it for free: reference the asset the way the scene does, author the
spawner's zero the way the spawner does, and read where the root body lands.
Composition is deterministic, so the answer on a laptop is the answer on the
GPU - this is a property of the USD file alone, not of the machine opening it.

Placement that survives lives on prims the spawner never touches - the bodies
below the default prim. This gate does not care how the asset achieved
immunity; it cares only that the root body lands where the spec says, both
with and without the spawner's opinion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


SPAWN_POSE_IMMUNITY_SCHEMA_VERSION = "spawn_pose_immunity.v1"
DEFAULT_TOLERANCE_M = 0.005


class SpawnPoseImmunityError(ValueError):
    """Stable, sorted immunity-probe failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _compose_root_body_position(
    asset_path: Path, root_body_prim_name: str, *, author_spawner_zero: bool
) -> tuple[float, float, float]:
    """Reference the asset and read the root body's composed world position."""

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/task_object", "Xform")
    prim.GetReferences().AddReference(str(asset_path))

    if author_spawner_zero:
        # What sim_utils does at scene construction: set local opinions for
        # translate and orient on the prim it just created. Setting through
        # the resolved ops writes to the local layer, which is exactly the
        # opinion strength the spawner's write has.
        xformable = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op for op in xformable.GetOrderedXformOps()}
        translate = ops.get(UsdGeom.XformOp.TypeTranslate)
        if translate is None:
            translate = xformable.AddTranslateOp()
        translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        orient = ops.get(UsdGeom.XformOp.TypeOrient)
        if orient is not None:
            orient.Set(type(orient.Get())(1.0, 0.0, 0.0, 0.0))

    body = stage.GetPrimAtPath(f"/World/task_object/{root_body_prim_name}")
    if not body:
        raise SpawnPoseImmunityError(
            [f"spawn_pose_immunity_root_body_prim_missing:{root_body_prim_name}"]
        )
    translation = (
        UsdGeom.XformCache().GetLocalToWorldTransform(body).ExtractTranslation()
    )
    return (float(translation[0]), float(translation[1]), float(translation[2]))


def probe_spawn_pose_immunity(
    *,
    asset_path: str | Path,
    root_body_prim_name: str,
    expected_world_position_m: Sequence[float],
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> dict[str, Any]:
    """Prove the asset lands at its pose even after the spawner's zero-set."""

    path = Path(asset_path).expanduser()
    if not path.is_file():
        raise SpawnPoseImmunityError([f"spawn_pose_immunity_asset_missing:{path}"])
    expected = tuple(float(v) for v in expected_world_position_m)
    if len(expected) != 3:
        raise SpawnPoseImmunityError(
            [f"spawn_pose_immunity_expected_position_not_3d:{len(expected)}"]
        )

    unauthored = _compose_root_body_position(
        path, root_body_prim_name, author_spawner_zero=False
    )
    after_zero = _compose_root_body_position(
        path, root_body_prim_name, author_spawner_zero=True
    )

    errors: list[str] = []
    drift = max(abs(a - b) for a, b in zip(unauthored, after_zero))
    if drift > float(tolerance_m):
        errors.append(
            "spawn_pose_immunity_spawn_zero_defeats_placement:"
            f"moved_{drift:.4f}m_when_spawner_authored_zero"
        )
    miss = max(abs(a - b) for a, b in zip(after_zero, expected))
    if miss > float(tolerance_m):
        errors.append(
            "spawn_pose_immunity_expected_position_not_reached:"
            f"off_by_{miss:.4f}m_after_spawner_zero"
        )
    if errors:
        raise SpawnPoseImmunityError(errors)

    return {
        "schema_version": SPAWN_POSE_IMMUNITY_SCHEMA_VERSION,
        "asset_path": str(path),
        "root_body_prim_name": root_body_prim_name,
        "expected_world_position_m": list(expected),
        "position_unauthored_m": list(unauthored),
        "position_after_spawner_zero_m": list(after_zero),
        "tolerance_m": float(tolerance_m),
        "immune": True,
        "claim_boundary": {
            "composition_on_the_laptop_is_composition_on_the_gpu": True,
            "immunity_is_to_the_spawner_zero_set_not_to_physics": True,
        },
    }


__all__ = [
    "DEFAULT_TOLERANCE_M",
    "SPAWN_POSE_IMMUNITY_SCHEMA_VERSION",
    "SpawnPoseImmunityError",
    "probe_spawn_pose_immunity",
]
