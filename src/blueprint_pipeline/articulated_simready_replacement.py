"""Static qualification contracts for the articulated SimReady replacement.

The Joint Agent owned-core asset is an articulation-topology candidate, never a
qualified SimReady object. This module deterministically admits or rejects the
Blueprint-authored articulated replacement against the frozen scene/task
contract before any native simulator time is spent. Every check is hermetic:
it reads only the replacement USD bytes and the frozen contract values.

Claim separation is explicit: passing these gates yields a statically admitted
candidate. It never asserts native-simulator qualification or physical
equivalence, and generated interior geometry remains a candidate, not observed
site truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


TOPOLOGY_SCHEMA_VERSION = "articulated_replacement_topology_validation.v1"
PHYSICS_SCHEMA_VERSION = "articulated_replacement_physics_validation.v1"

# Deterministic provenance labels. Every geometry prim in the replacement must
# declare whether it derives from retained source observations or is generated
# candidate geometry (for example the never-observed interior). An untagged
# prim fails closed so generated surfaces can never silently read as observed
# site truth.
PROVENANCE_ATTRIBUTE = "blueprint:articulatedReplacement:provenance"
OBSERVED_PROVENANCE_VALUE = "observed_source_derived"
GENERATED_PROVENANCE_VALUE = "generated_candidate_geometry"
TASK_CONTACT_ROLE_ATTRIBUTE = "blueprint:articulatedReplacement:taskContactRole"
HANDLE_ROLE_VALUE = "handle"


class ArticulatedSimReadyReplacementError(ValueError):
    """Stable, sorted articulated-replacement qualification failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_vector(value: Any, length: int) -> tuple[float, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        return None
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return tuple(result)


def _matrix4(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    rows: list[tuple[float, ...]] = []
    for row in value:
        vector = _finite_vector(row, 4)
        if vector is None:
            return None
        rows.append(vector)
    return tuple(rows)


def _transform_point(
    matrix: Sequence[Sequence[float]], point: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        matrix[axis][0] * point[0]
        + matrix[axis][1] * point[1]
        + matrix[axis][2] * point[2]
        + matrix[axis][3]
        for axis in range(3)
    )  # type: ignore[return-value]


def _rotate_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(
        matrix[axis][0] * vector[0]
        + matrix[axis][1] * vector[1]
        + matrix[axis][2] * vector[2]
        for axis in range(3)
    )  # type: ignore[return-value]


def _normalized(vector: Sequence[float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        return None
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _validated_contract(contract: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    task_joint_id = str(contract.get("task_joint_id") or "")
    if not task_joint_id:
        errors.append("articulated_replacement_contract_task_joint_id_missing")
    resolved["task_joint_id"] = task_joint_id

    hinge_world = _finite_vector(contract.get("hinge_origin_world_m"), 3)
    if hinge_world is None:
        errors.append("articulated_replacement_contract_hinge_origin_invalid")
    matrix = _matrix4(contract.get("T_asset_world"))
    if matrix is None:
        errors.append("articulated_replacement_contract_asset_transform_invalid")
    if hinge_world is not None and matrix is not None:
        resolved["hinge_asset_m"] = _transform_point(matrix, hinge_world)
    axis_world = _finite_vector(contract.get("task_axis_world"), 3)
    axis_asset = None
    if axis_world is not None and matrix is not None:
        axis_asset = _normalized(_rotate_vector(matrix, axis_world))
    if axis_asset is None:
        errors.append("articulated_replacement_contract_task_axis_invalid")
    resolved["task_axis_asset"] = axis_asset

    axis_dot_minimum = contract.get("task_axis_absolute_dot_minimum")
    if (
        isinstance(axis_dot_minimum, bool)
        or not isinstance(axis_dot_minimum, (int, float))
        or not 0.0 < float(axis_dot_minimum) <= 1.0
    ):
        errors.append("articulated_replacement_contract_axis_threshold_invalid")
        axis_dot_minimum = 2.0
    resolved["axis_dot_minimum"] = float(axis_dot_minimum)

    interval = _finite_vector(contract.get("task_moving_z_interval_m"), 2)
    if interval is None or interval[0] >= interval[1]:
        errors.append("articulated_replacement_contract_task_interval_invalid")
        interval = (math.inf, -math.inf)
    resolved["task_interval"] = interval

    overlap_minimum = contract.get("task_z_overlap_minimum")
    if (
        isinstance(overlap_minimum, bool)
        or not isinstance(overlap_minimum, (int, float))
        or not 0.0 < float(overlap_minimum) <= 1.0
    ):
        errors.append("articulated_replacement_contract_overlap_threshold_invalid")
        overlap_minimum = 2.0
    resolved["overlap_minimum"] = float(overlap_minimum)

    limits = _finite_vector(contract.get("task_limits_rad"), 2)
    if limits is None or limits[0] >= limits[1]:
        errors.append("articulated_replacement_contract_task_limits_invalid")
        limits = (math.nan, math.nan)
    resolved["task_limits_rad"] = limits

    limits_tolerance = contract.get("limits_tolerance_rad")
    if (
        isinstance(limits_tolerance, bool)
        or not isinstance(limits_tolerance, (int, float))
        or float(limits_tolerance) < 0.0
    ):
        errors.append("articulated_replacement_contract_limits_tolerance_invalid")
        limits_tolerance = -1.0
    resolved["limits_tolerance_rad"] = float(limits_tolerance)

    pivot_tolerance = contract.get("pivot_xy_tolerance_m")
    if (
        isinstance(pivot_tolerance, bool)
        or not isinstance(pivot_tolerance, (int, float))
        or float(pivot_tolerance) <= 0.0
    ):
        errors.append("articulated_replacement_contract_pivot_tolerance_invalid")
        pivot_tolerance = -1.0
    resolved["pivot_xy_tolerance_m"] = float(pivot_tolerance)

    minimum_joints = contract.get("minimum_assembly_joint_count")
    maximum_joints = contract.get("maximum_assembly_joint_count")
    if (
        isinstance(minimum_joints, bool)
        or isinstance(maximum_joints, bool)
        or not isinstance(minimum_joints, int)
        or not isinstance(maximum_joints, int)
        or minimum_joints < 1
        or maximum_joints < minimum_joints
    ):
        errors.append("articulated_replacement_contract_joint_scope_invalid")
        minimum_joints, maximum_joints = 1, 0
    resolved["minimum_joints"] = minimum_joints
    resolved["maximum_joints"] = maximum_joints

    required_roots = contract.get("required_articulation_root_count")
    if isinstance(required_roots, bool) or not isinstance(required_roots, int) or required_roots < 1:
        errors.append("articulated_replacement_contract_root_count_invalid")
        required_roots = -1
    resolved["required_roots"] = required_roots
    return resolved


def validate_articulated_replacement_topology(
    *,
    replacement_usd_path: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit the replacement's joint graph against the frozen task contract.

    The validator resolves the task joint from geometry and axis evidence the
    same way the in-run Joint Agent review does: a prim name or authored label
    alone can never select the commanded joint.
    """

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedSimReadyReplacementError(
            ["articulated_replacement_openusd_runtime_missing"]
        ) from exc

    errors: list[str] = []
    resolved = _validated_contract(contract, errors)

    path = Path(replacement_usd_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArticulatedSimReadyReplacementError(
            ["articulated_replacement_usd_missing"]
        )
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ArticulatedSimReadyReplacementError(
            ["articulated_replacement_usd_unreadable"]
        )

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-9:
        errors.append("articulated_replacement_meters_per_unit_not_one")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("articulated_replacement_up_axis_not_z")

    roots = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    if resolved.get("required_roots", -1) >= 1 and len(roots) != resolved["required_roots"]:
        errors.append(
            "articulated_replacement_articulation_root_count_mismatch:"
            f"observed={len(roots)}"
        )

    joint_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    minimum_joints = resolved.get("minimum_joints", 1)
    maximum_joints = resolved.get("maximum_joints", 0)
    if not minimum_joints <= len(joint_prims) <= maximum_joints:
        errors.append(
            "articulated_replacement_assembly_joint_count_outside_preregistered_bounds:"
            f"observed={len(joint_prims)}"
        )

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    axis_tokens = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    task_axis = resolved.get("task_axis_asset")
    task_interval = resolved.get("task_interval", (math.inf, -math.inf))
    hinge_asset = resolved.get("hinge_asset_m")

    rows: list[dict[str, Any]] = []
    task_matches: list[str] = []
    for prim in sorted(joint_prims, key=lambda item: str(item.GetPath())):
        joint = UsdPhysics.Joint(prim)
        joint_path = str(prim.GetPath())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
        else:
            joint_type = "unsupported"
            errors.append(
                f"articulated_replacement_joint_type_not_admitted:{joint_path}"
            )
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            errors.append(
                f"articulated_replacement_joint_bodies_unresolved:{joint_path}"
            )
            continue
        body0 = stage.GetPrimAtPath(body0_targets[0])
        body1 = stage.GetPrimAtPath(body1_targets[0])
        if not body0.IsValid() or not body1.IsValid():
            errors.append(
                f"articulated_replacement_joint_body_prim_missing:{joint_path}"
            )
            continue

        body0_world = xform_cache.GetLocalToWorldTransform(body0)
        local_pos0 = joint.GetLocalPos0Attr().Get()
        local_rot0 = joint.GetLocalRot0Attr().Get()
        pivot_local = (
            (float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2]))
            if local_pos0 is not None
            else (0.0, 0.0, 0.0)
        )
        pivot_world_gf = body0_world.Transform(Gf.Vec3d(*pivot_local))
        pivot_asset = (
            float(pivot_world_gf[0]),
            float(pivot_world_gf[1]),
            float(pivot_world_gf[2]),
        )

        axis_token = None
        axis_attr = prim.GetAttribute("physics:axis")
        if axis_attr and axis_attr.HasAuthoredValue():
            axis_token = str(axis_attr.Get())
        axis_local = axis_tokens.get(axis_token or "")
        axis_dot = None
        if axis_local is not None and task_axis is not None:
            rotation = body0_world.ExtractRotationMatrix()
            if local_rot0 is not None:
                rotation = Gf.Matrix3d(Gf.Rotation(local_rot0)) * rotation
            axis_vec = Gf.Vec3d(*axis_local) * rotation
            normalized = _normalized((axis_vec[0], axis_vec[1], axis_vec[2]))
            if normalized is not None:
                axis_dot = abs(
                    sum(left * right for left, right in zip(normalized, task_axis))
                )
        if axis_dot is None and joint_type == "revolute":
            errors.append(
                f"articulated_replacement_joint_axis_unresolved:{joint_path}"
            )

        moving_range = cache.ComputeWorldBound(body1).ComputeAlignedRange()
        if moving_range.IsEmpty():
            errors.append(
                f"articulated_replacement_moving_link_bounds_empty:{joint_path}"
            )
            overlap_fraction = None
            moving_interval = None
        else:
            low = float(moving_range.GetMin()[2])
            high = float(moving_range.GetMax()[2])
            moving_interval = (low, high)
            span = task_interval[1] - task_interval[0]
            overlap = max(
                0.0, min(high, task_interval[1]) - max(low, task_interval[0])
            )
            overlap_fraction = overlap / span if span > 0 else None

        lower_limit_deg = None
        upper_limit_deg = None
        if joint_type == "revolute":
            revolute = UsdPhysics.RevoluteJoint(prim)
            lower_attr = revolute.GetLowerLimitAttr().Get()
            upper_attr = revolute.GetUpperLimitAttr().Get()
            lower_limit_deg = float(lower_attr) if lower_attr is not None else None
            upper_limit_deg = float(upper_attr) if upper_attr is not None else None

        matches = (
            joint_type == "revolute"
            and axis_dot is not None
            and axis_dot >= resolved.get("axis_dot_minimum", 2.0)
            and overlap_fraction is not None
            and overlap_fraction >= resolved.get("overlap_minimum", 2.0)
        )
        if matches:
            task_matches.append(joint_path)
            limits = resolved.get("task_limits_rad", (math.nan, math.nan))
            tolerance = resolved.get("limits_tolerance_rad", -1.0)
            if lower_limit_deg is None or upper_limit_deg is None:
                errors.append(
                    f"articulated_replacement_task_joint_limits_missing:{joint_path}"
                )
            else:
                lower_rad = math.radians(lower_limit_deg)
                upper_rad = math.radians(upper_limit_deg)
                if (
                    not math.isfinite(limits[0])
                    or tolerance < 0.0
                    or abs(lower_rad - limits[0]) > tolerance
                    or abs(upper_rad - limits[1]) > tolerance
                ):
                    errors.append(
                        "articulated_replacement_task_joint_limits_mismatch:"
                        f"{joint_path}:lower_rad={lower_rad!r}:upper_rad={upper_rad!r}"
                    )
            if hinge_asset is not None:
                pivot_tolerance = resolved.get("pivot_xy_tolerance_m", -1.0)
                planar = math.hypot(
                    pivot_asset[0] - hinge_asset[0], pivot_asset[1] - hinge_asset[1]
                )
                if pivot_tolerance < 0.0 or planar > pivot_tolerance:
                    errors.append(
                        "articulated_replacement_task_joint_pivot_outside_tolerance:"
                        f"{joint_path}:planar_m={planar!r}"
                    )

        rows.append(
            {
                "joint_prim_path": joint_path,
                "joint_type": joint_type,
                "body0": str(body0_targets[0]),
                "body1": str(body1_targets[0]),
                "pivot_asset_m": list(pivot_asset),
                "axis_absolute_dot": axis_dot,
                "moving_z_interval_m": list(moving_interval) if moving_interval else None,
                "target_z_overlap_fraction": overlap_fraction,
                "lower_limit_deg": lower_limit_deg,
                "upper_limit_deg": upper_limit_deg,
                "task_match": matches,
            }
        )

    if len(task_matches) != 1:
        errors.append(
            "articulated_replacement_exactly_one_task_joint_not_resolved:"
            f"matches={len(task_matches)}"
        )

    if errors:
        raise ArticulatedSimReadyReplacementError(errors)

    receipt: dict[str, Any] = {
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "status": "topology_statically_admitted",
        "replacement_usd_path": str(path),
        "replacement_usd_sha256": _sha256(path),
        "task_joint_id": resolved["task_joint_id"],
        "task_joint_prim_path": task_matches[0],
        "non_task_joint_prim_paths": sorted(
            row["joint_prim_path"] for row in rows if not row["task_match"]
        ),
        "articulation_root_prim_paths": roots,
        "assembly_joint_count": len(rows),
        "joint_review": rows,
        "claim_boundary": {
            "topology_admission_is_not_native_qualification": True,
            "native_simulator_qualified": False,
            "physical_equivalence_proven": False,
            "generated_geometry_is_observed_site_truth": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _range_check(
    value: Any, bounds: Any, *, errors: list[str], error: str
) -> float | None:
    number = (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )
    interval = _finite_vector(bounds, 2)
    if interval is None:
        errors.append(f"{error}_contract_range_invalid")
        return number
    if number is None or not math.isfinite(number) or not interval[0] <= number <= interval[1]:
        errors.append(error)
    return number


def _aabb_overlap_depth(
    a_min: Sequence[float],
    a_max: Sequence[float],
    b_min: Sequence[float],
    b_max: Sequence[float],
) -> float:
    depth = math.inf
    for axis in range(3):
        overlap = min(a_max[axis], b_max[axis]) - max(a_min[axis], b_min[axis])
        if overlap <= 0.0:
            return 0.0
        depth = min(depth, overlap)
    return depth


def validate_articulated_replacement_physics(
    *,
    replacement_usd_path: str | Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Statically admit rigid bodies, colliders, materials, and provenance.

    This gate is deliberately conservative and purely static: axis-aligned
    bounds stand in for narrowphase contact, so passing it never claims
    native contact stability. Its job is to fail closed on structure a native
    run could silently tolerate: colliders spanning the door seam, unmassed
    links, floating bodies, missing handle contact geometry, missing generated
    interior, untagged provenance, or an unsupported cabinet.
    """

    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedSimReadyReplacementError(
            ["articulated_replacement_openusd_runtime_missing"]
        ) from exc

    errors: list[str] = []
    path = Path(replacement_usd_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ArticulatedSimReadyReplacementError(["articulated_replacement_usd_missing"])
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ArticulatedSimReadyReplacementError(["articulated_replacement_usd_unreadable"])

    envelopes = contract.get("link_collider_envelopes_m")
    if not isinstance(envelopes, Mapping) or not envelopes:
        errors.append("articulated_replacement_contract_link_envelopes_invalid")
        envelopes = {}
    task_door_link = str(contract.get("task_door_link") or "")
    if not task_door_link:
        errors.append("articulated_replacement_contract_task_door_link_missing")
    handle_protrusion = contract.get("handle_minimum_protrusion_m")
    if (
        isinstance(handle_protrusion, bool)
        or not isinstance(handle_protrusion, (int, float))
        or float(handle_protrusion) <= 0.0
    ):
        errors.append("articulated_replacement_contract_handle_protrusion_invalid")
        handle_protrusion = math.inf
    max_overlap = contract.get("maximum_reset_pairwise_overlap_m")
    if (
        isinstance(max_overlap, bool)
        or not isinstance(max_overlap, (int, float))
        or float(max_overlap) < 0.0
    ):
        errors.append("articulated_replacement_contract_reset_overlap_invalid")
        max_overlap = -1.0
    support_tolerance = contract.get("support_z_tolerance_m")
    if (
        isinstance(support_tolerance, bool)
        or not isinstance(support_tolerance, (int, float))
        or float(support_tolerance) <= 0.0
    ):
        errors.append("articulated_replacement_contract_support_tolerance_invalid")
        support_tolerance = -1.0
    required_interior_links = contract.get("required_generated_interior_links")
    if not isinstance(required_interior_links, list) or not required_interior_links:
        errors.append("articulated_replacement_contract_interior_links_invalid")
        required_interior_links = []

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    rigid_links = sorted(
        (
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
    )
    joint_bodies: set[str] = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for target in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()):
            joint_bodies.add(str(target))
    for link_path in rigid_links:
        if link_path not in joint_bodies:
            errors.append(
                f"articulated_replacement_floating_rigid_body_outside_joint_graph:{link_path}"
            )

    link_rows: list[dict[str, Any]] = []
    collider_bounds_by_link: dict[str, list[tuple[str, list[float], list[float]]]] = {}
    for link_path in rigid_links:
        prim = stage.GetPrimAtPath(link_path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_value = mass_api.GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        mass = _range_check(
            mass_value,
            contract.get("mass_range_kg"),
            errors=errors,
            error=f"articulated_replacement_link_mass_missing_or_out_of_range:{link_path}",
        ) if mass_value is not None else None
        if mass_value is None:
            errors.append(f"articulated_replacement_link_mass_missing:{link_path}")
        center_of_mass = (
            mass_api.GetCenterOfMassAttr().Get()
            if prim.HasAPI(UsdPhysics.MassAPI)
            else None
        )
        inertia = (
            mass_api.GetDiagonalInertiaAttr().Get()
            if prim.HasAPI(UsdPhysics.MassAPI)
            else None
        )
        link_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        link_min = [float(link_range.GetMin()[axis]) for axis in range(3)]
        link_max = [float(link_range.GetMax()[axis]) for axis in range(3)]
        if center_of_mass is None:
            errors.append(f"articulated_replacement_link_center_of_mass_missing:{link_path}")
        else:
            com = [float(center_of_mass[axis]) for axis in range(3)]
            if any(com[axis] < link_min[axis] - 1e-6 or com[axis] > link_max[axis] + 1e-6 for axis in range(3)):
                errors.append(
                    f"articulated_replacement_link_center_of_mass_outside_bounds:{link_path}"
                )
        if inertia is None or any(float(inertia[axis]) <= 0.0 for axis in range(3)):
            errors.append(
                f"articulated_replacement_link_inertia_missing_or_non_positive:{link_path}"
            )
        link_rows.append(
            {
                "link_prim_path": link_path,
                "mass_kg": float(mass_value) if mass_value is not None else None,
                "aabb_min_m": link_min,
                "aabb_max_m": link_max,
            }
        )
        del mass

    friction_range = contract.get("friction_range")
    restitution_range = contract.get("restitution_range")
    handle_paths: list[str] = []
    generated_interior_by_link: dict[str, list[str]] = {}
    untagged: list[str] = []
    collider_rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        is_geometry = prim.IsA(UsdGeom.Gprim)
        if not is_geometry:
            continue
        prim_path = str(prim.GetPath())
        provenance_attr = prim.GetAttribute(PROVENANCE_ATTRIBUTE)
        provenance = (
            str(provenance_attr.Get())
            if provenance_attr and provenance_attr.HasAuthoredValue()
            else None
        )
        if provenance not in {OBSERVED_PROVENANCE_VALUE, GENERATED_PROVENANCE_VALUE}:
            untagged.append(prim_path)
        owning_link = None
        parent = prim.GetParent()
        while parent and parent.GetPath() != prim.GetStage().GetPseudoRoot().GetPath():
            if str(parent.GetPath()) in set(rigid_links):
                owning_link = str(parent.GetPath())
                break
            parent = parent.GetParent()
        if provenance == GENERATED_PROVENANCE_VALUE and owning_link is not None:
            generated_interior_by_link.setdefault(owning_link, []).append(prim_path)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if owning_link is None:
            errors.append(
                f"articulated_replacement_collider_without_rigid_link:{prim_path}"
            )
            continue
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bound.IsEmpty():
            errors.append(f"articulated_replacement_collider_bounds_empty:{prim_path}")
            continue
        bound_min = [float(bound.GetMin()[axis]) for axis in range(3)]
        bound_max = [float(bound.GetMax()[axis]) for axis in range(3)]
        collider_bounds_by_link.setdefault(owning_link, []).append(
            (prim_path, bound_min, bound_max)
        )
        envelope = envelopes.get(owning_link)
        if isinstance(envelope, Mapping):
            envelope_min = _finite_vector(envelope.get("aabb_min"), 3)
            envelope_max = _finite_vector(envelope.get("aabb_max"), 3)
            if envelope_min is None or envelope_max is None:
                errors.append(
                    f"articulated_replacement_contract_link_envelope_invalid:{owning_link}"
                )
            elif any(
                bound_min[axis] < envelope_min[axis] or bound_max[axis] > envelope_max[axis]
                for axis in range(3)
            ):
                errors.append(
                    f"articulated_replacement_collider_outside_link_envelope:{prim_path}"
                )
        else:
            errors.append(
                f"articulated_replacement_link_envelope_missing:{owning_link}"
            )
        role_attr = prim.GetAttribute(TASK_CONTACT_ROLE_ATTRIBUTE)
        role = str(role_attr.Get()) if role_attr and role_attr.HasAuthoredValue() else None
        if role == HANDLE_ROLE_VALUE and owning_link == task_door_link:
            handle_paths.append(prim_path)
        material_bound = False
        binding_api = UsdShade.MaterialBindingAPI(prim)
        bound_material, _relationship = binding_api.ComputeBoundMaterial(
            materialPurpose="physics"
        )
        material_prim = bound_material.GetPrim() if bound_material else None
        if (
            material_prim is not None
            and material_prim.IsValid()
            and material_prim.HasAPI(UsdPhysics.MaterialAPI)
        ):
            material_api = UsdPhysics.MaterialAPI(material_prim)
            _range_check(
                material_api.GetStaticFrictionAttr().Get(),
                friction_range,
                errors=errors,
                error=f"articulated_replacement_static_friction_out_of_range:{prim_path}",
            )
            _range_check(
                material_api.GetDynamicFrictionAttr().Get(),
                friction_range,
                errors=errors,
                error=f"articulated_replacement_dynamic_friction_out_of_range:{prim_path}",
            )
            _range_check(
                material_api.GetRestitutionAttr().Get(),
                restitution_range,
                errors=errors,
                error=f"articulated_replacement_restitution_out_of_range:{prim_path}",
            )
            material_bound = True
        if not material_bound:
            errors.append(
                f"articulated_replacement_collider_physics_material_missing:{prim_path}"
            )
        collider_rows.append(
            {
                "collider_prim_path": prim_path,
                "link_prim_path": owning_link,
                "aabb_min_m": bound_min,
                "aabb_max_m": bound_max,
                "task_contact_role": role,
                "provenance": provenance,
            }
        )

    for link_path in rigid_links:
        if not collider_bounds_by_link.get(link_path):
            errors.append(
                f"articulated_replacement_link_collider_missing:{link_path}"
            )

    link_paths = sorted(collider_bounds_by_link)
    for left_index in range(len(link_paths)):
        for right_index in range(left_index + 1, len(link_paths)):
            left_link = link_paths[left_index]
            right_link = link_paths[right_index]
            for left_path, left_min, left_max in collider_bounds_by_link[left_link]:
                for right_path, right_min, right_max in collider_bounds_by_link[right_link]:
                    depth = _aabb_overlap_depth(left_min, left_max, right_min, right_max)
                    if max_overlap >= 0.0 and depth > max_overlap:
                        errors.append(
                            "articulated_replacement_reset_pose_collider_penetration:"
                            f"{left_path}:{right_path}:depth_m={depth!r}"
                        )

    support_link = str(contract.get("support_link") or "")
    if not support_link:
        errors.append("articulated_replacement_contract_support_link_missing")
    elif support_tolerance > 0.0:
        support_bounds = collider_bounds_by_link.get(support_link) or []
        if not support_bounds:
            errors.append(
                f"articulated_replacement_support_link_collider_missing:{support_link}"
            )
        else:
            minimum_z = min(bounds[1][2] for bounds in support_bounds)
            if abs(minimum_z) > support_tolerance:
                errors.append(
                    "articulated_replacement_support_contact_not_grounded:"
                    f"{support_link}:minimum_z_m={minimum_z!r}"
                )

    if task_door_link and not handle_paths:
        errors.append(
            f"articulated_replacement_task_door_handle_contact_missing:{task_door_link}"
        )
    else:
        door_bounds = collider_bounds_by_link.get(task_door_link) or []
        door_front = max((bounds[2][1] for bounds in door_bounds), default=None)
        for handle_path in handle_paths:
            handle_bounds = next(
                (
                    bounds
                    for bounds in door_bounds
                    if bounds[0] == handle_path
                ),
                None,
            )
            if handle_bounds is None or door_front is None:
                continue
            non_handle_front = max(
                (
                    bounds[2][1]
                    for bounds in door_bounds
                    if bounds[0] not in set(handle_paths)
                ),
                default=None,
            )
            if (
                non_handle_front is not None
                and handle_bounds[2][1] - non_handle_front < float(handle_protrusion)
            ):
                errors.append(
                    f"articulated_replacement_handle_protrusion_insufficient:{handle_path}"
                )

    for link_path in required_interior_links:
        if not generated_interior_by_link.get(str(link_path)):
            errors.append(
                f"articulated_replacement_generated_interior_missing:{link_path}"
            )

    for prim_path in sorted(untagged):
        errors.append(f"articulated_replacement_geometry_provenance_untagged:{prim_path}")

    if errors:
        raise ArticulatedSimReadyReplacementError(errors)

    receipt: dict[str, Any] = {
        "schema_version": PHYSICS_SCHEMA_VERSION,
        "status": "physics_statically_admitted",
        "replacement_usd_path": str(path),
        "replacement_usd_sha256": _sha256(path),
        "link_review": link_rows,
        "collider_review": collider_rows,
        "handle_prim_paths": sorted(set(handle_paths)),
        "generated_interior_prim_paths": sorted(
            prim_path
            for paths in generated_interior_by_link.values()
            for prim_path in paths
        ),
        "claim_boundary": {
            "static_admission_is_not_contact_stability": True,
            "native_simulator_qualified": False,
            "physical_equivalence_proven": False,
            "generated_geometry_is_observed_site_truth": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


AUTHORING_SCHEMA_VERSION = "articulated_simready_authoring.v1"


def _local_range(cache: Any, prim: Any) -> tuple[list[float], list[float]]:
    aligned = cache.ComputeLocalBound(prim).ComputeAlignedRange()
    return (
        [float(aligned.GetMin()[axis]) for axis in range(3)],
        [float(aligned.GetMax()[axis]) for axis in range(3)],
    )


def author_articulated_simready_replacement(
    *,
    rigged_topology_usd_path: str | Path,
    output_usd_path: str | Path,
    topology_contract: Mapping[str, Any],
    physics_contract_template: Mapping[str, Any],
    authoring_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically author physics onto an owned-core topology candidate.

    Joint Agent's owned-core output is topology-only (joints, no masses or
    colliders). This function adds the Blueprint-owned physics: rigid bodies,
    density-independent role-based masses with box-approximation inertia,
    per-component convex-hull colliders, an inherited physics material,
    handle contact geometry on the task door (observed component when one
    protrudes, otherwise a generated parametric candidate), a generated
    interior block behind the doors, and observed/generated provenance tags.
    The output is admitted only if both static validators pass; nothing here
    is native-simulator qualification or physical equivalence.
    """

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ArticulatedSimReadyReplacementError(
            ["articulated_replacement_openusd_runtime_missing"]
        ) from exc

    rigged = Path(rigged_topology_usd_path).expanduser().resolve()
    output = Path(output_usd_path).expanduser().resolve()
    if not rigged.is_file() or rigged.is_symlink():
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_rigged_input_missing"]
        )
    spec = json.loads(json.dumps(dict(authoring_spec)))
    for field in (
        "support_link_mass_kg",
        "door_link_mass_kg",
        "other_link_mass_kg",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "generated_interior_inset_m",
    ):
        value = spec.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise ArticulatedSimReadyReplacementError(
                [f"articulated_authoring_spec_invalid:{field}"]
            )
    handle_spec = spec.get("handle")
    if (
        not isinstance(handle_spec, Mapping)
        or _finite_vector(handle_spec.get("generated_center_asset_m"), 3) is None
        or _finite_vector(handle_spec.get("generated_half_extents_m"), 3) is None
        or not isinstance(handle_spec.get("minimum_protrusion_m"), (int, float))
        or isinstance(handle_spec.get("minimum_protrusion_m"), bool)
        or float(handle_spec.get("minimum_protrusion_m")) <= 0.0
    ):
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_spec_invalid:handle"]
        )

    source_stage = Usd.Stage.Open(str(rigged))
    if source_stage is None:
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_rigged_input_unreadable"]
        )
    flat_layer = source_stage.Flatten()
    # Flattened layers embed the absolute source path in their documentation;
    # scrub identity so byte determinism depends only on content.
    flat_layer.documentation = ""
    flat_layer.comment = ""
    stage = Usd.Stage.Open(flat_layer)
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_default_prim_missing"]
        )

    resolved = _validated_contract(dict(topology_contract), [])
    task_axis = resolved.get("task_axis_asset")
    task_interval = resolved.get("task_interval", (math.inf, -math.inf))
    axis_tokens = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

    joint_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    if not joint_prims:
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_topology_joints_missing"]
        )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    joint_rows: list[dict[str, Any]] = []
    for prim in sorted(joint_prims, key=lambda item: str(item.GetPath())):
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            raise ArticulatedSimReadyReplacementError(
                [f"articulated_authoring_joint_bodies_unresolved:{prim.GetPath()}"]
            )
        body1 = stage.GetPrimAtPath(body1_targets[0])
        if not body1.IsValid():
            raise ArticulatedSimReadyReplacementError(
                [f"articulated_authoring_joint_body_missing:{prim.GetPath()}"]
            )
        axis_attr = prim.GetAttribute("physics:axis")
        axis_local = axis_tokens.get(
            str(axis_attr.Get()) if axis_attr and axis_attr.HasAuthoredValue() else ""
        )
        axis_dot = None
        if axis_local is not None and task_axis is not None:
            axis_dot = abs(
                sum(left * right for left, right in zip(axis_local, task_axis))
            )
        moving_range = cache.ComputeWorldBound(body1).ComputeAlignedRange()
        overlap_fraction = None
        if not moving_range.IsEmpty():
            low = float(moving_range.GetMin()[2])
            high = float(moving_range.GetMax()[2])
            span = task_interval[1] - task_interval[0]
            if span > 0:
                overlap_fraction = max(
                    0.0, min(high, task_interval[1]) - max(low, task_interval[0])
                ) / span
        joint_rows.append(
            {
                "prim": prim,
                "path": str(prim.GetPath()),
                "body0": str(body0_targets[0]),
                "body1": str(body1_targets[0]),
                "task_match": bool(
                    prim.IsA(UsdPhysics.RevoluteJoint)
                    and axis_dot is not None
                    and axis_dot >= resolved.get("axis_dot_minimum", 2.0)
                    and overlap_fraction is not None
                    and overlap_fraction >= resolved.get("overlap_minimum", 2.0)
                ),
            }
        )
    task_joints = [row for row in joint_rows if row["task_match"]]
    if len(task_joints) != 1:
        raise ArticulatedSimReadyReplacementError(
            [
                "articulated_authoring_exactly_one_task_joint_not_resolved:"
                f"matches={len(task_joints)}"
            ]
        )
    task_joint = task_joints[0]
    support_link_path = task_joint["body0"]
    task_door_path = task_joint["body1"]
    link_paths = sorted(
        {row["body0"] for row in joint_rows} | {row["body1"] for row in joint_rows}
    )
    door_link_paths = sorted({row["body1"] for row in joint_rows})

    # Exactly one articulation root: the default prim.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim != default_prim:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(default_prim)

    # Frozen task limits on every revolute joint, matching the freeze contract.
    limits = resolved.get("task_limits_rad", (math.nan, math.nan))
    for row in joint_rows:
        prim = row["prim"]
        if prim.IsA(UsdPhysics.RevoluteJoint) and math.isfinite(limits[0]):
            revolute = UsdPhysics.RevoluteJoint(prim)
            revolute.CreateLowerLimitAttr().Set(math.degrees(limits[0]))
            revolute.CreateUpperLimitAttr().Set(math.degrees(limits[1]))

    material = UsdShade.Material.Define(
        stage, f"{default_prim.GetPath()}/physics_materials/replacement_default"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(float(spec["static_friction"]))
    material_api.CreateDynamicFrictionAttr().Set(float(spec["dynamic_friction"]))
    material_api.CreateRestitutionAttr().Set(float(spec["restitution"]))
    UsdShade.MaterialBindingAPI.Apply(default_prim).Bind(
        material, materialPurpose="physics"
    )

    def _tag(prim: Any, provenance: str) -> None:
        prim.CreateAttribute(PROVENANCE_ATTRIBUTE, Sdf.ValueTypeNames.String).Set(
            provenance
        )

    link_rows: list[dict[str, Any]] = []
    for link_path in link_paths:
        prim = stage.GetPrimAtPath(link_path)
        if not prim.IsValid():
            raise ArticulatedSimReadyReplacementError(
                [f"articulated_authoring_link_missing:{link_path}"]
            )
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if link_path == support_link_path:
            mass = float(spec["support_link_mass_kg"])
        elif link_path in door_link_paths:
            mass = float(spec["door_link_mass_kg"])
        else:
            mass = float(spec["other_link_mass_kg"])
        low, high = _local_range(cache, prim)
        dims = [max(high[axis] - low[axis], 1e-4) for axis in range(3)]
        center = [(high[axis] + low[axis]) / 2.0 for axis in range(3)]
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(mass)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center))
        mass_api.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(
                mass * (dims[1] ** 2 + dims[2] ** 2) / 12.0,
                mass * (dims[0] ** 2 + dims[2] ** 2) / 12.0,
                mass * (dims[0] ** 2 + dims[1] ** 2) / 12.0,
            )
        )
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(child)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_collision.CreateApproximationAttr().Set("convexHull")
                _tag(child, OBSERVED_PROVENANCE_VALUE)
        link_rows.append(
            {
                "link_prim_path": link_path,
                "role": (
                    "support"
                    if link_path == support_link_path
                    else "door" if link_path in door_link_paths else "other"
                ),
                "mass_kg": mass,
                "local_aabb_min_m": low,
                "local_aabb_max_m": high,
            }
        )

    # Handle: prefer an observed protruding component on the task door.
    door_prim = stage.GetPrimAtPath(task_door_path)
    door_gprims = [
        child for child in Usd.PrimRange(door_prim) if child.IsA(UsdGeom.Gprim)
    ]
    fronts = {}
    for child in door_gprims:
        aligned = cache.ComputeWorldBound(child).ComputeAlignedRange()
        if not aligned.IsEmpty():
            fronts[str(child.GetPath())] = float(aligned.GetMax()[1])
    handle_source = None
    handle_path = None
    minimum_protrusion = float(handle_spec["minimum_protrusion_m"])
    if len(fronts) >= 2:
        best_path, best_front = max(fronts.items(), key=lambda item: (item[1], item[0]))
        slab_front = max(
            front for path, front in fronts.items() if path != best_path
        )
        if best_front - slab_front >= minimum_protrusion:
            handle_path = best_path
            handle_source = "observed_source_component"
            stage.GetPrimAtPath(best_path).CreateAttribute(
                TASK_CONTACT_ROLE_ATTRIBUTE, Sdf.ValueTypeNames.String
            ).Set(HANDLE_ROLE_VALUE)
    if handle_path is None:
        center = [float(v) for v in handle_spec["generated_center_asset_m"]]
        half = [float(v) for v in handle_spec["generated_half_extents_m"]]
        handle_mesh = UsdGeom.Mesh.Define(
            stage, f"{task_door_path}/generated_task_handle"
        )
        points = [
            Gf.Vec3f(
                center[0] + sx * half[0],
                center[1] + sy * half[1],
                center[2] + sz * half[2],
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
        handle_mesh.CreatePointsAttr(points)
        handle_mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        handle_mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
        )
        handle_prim = handle_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(handle_prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(handle_prim)
        mesh_collision.CreateApproximationAttr().Set("convexHull")
        handle_prim.CreateAttribute(
            TASK_CONTACT_ROLE_ATTRIBUTE, Sdf.ValueTypeNames.String
        ).Set(HANDLE_ROLE_VALUE)
        _tag(handle_prim, GENERATED_PROVENANCE_VALUE)
        handle_path = str(handle_prim.GetPath())
        handle_source = "generated_parametric_candidate"

    # Generated interior: a solid inset block behind the doors so an opened
    # door never reveals a nonexistent shell. Candidate geometry, not truth.
    support_prim = stage.GetPrimAtPath(support_link_path)
    support_low, support_high = _local_range(cache, support_prim)
    inset = float(spec["generated_interior_inset_m"])
    interior_low = [support_low[axis] + inset for axis in range(3)]
    interior_high = [support_high[axis] - inset for axis in range(3)]
    if any(interior_low[axis] >= interior_high[axis] for axis in range(3)):
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_interior_inset_too_large"]
        )
    interior_mesh = UsdGeom.Mesh.Define(
        stage, f"{support_link_path}/generated_interior"
    )
    interior_points = [
        Gf.Vec3f(
            interior_low[0] if sx == 0 else interior_high[0],
            interior_low[1] if sy == 0 else interior_high[1],
            interior_low[2] if sz == 0 else interior_high[2],
        )
        for sx in (0, 1)
        for sy in (0, 1)
        for sz in (0, 1)
    ]
    interior_mesh.CreatePointsAttr(interior_points)
    interior_mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    interior_mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
    )
    _tag(interior_mesh.GetPrim(), GENERATED_PROVENANCE_VALUE)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output)):
        raise ArticulatedSimReadyReplacementError(
            ["articulated_authoring_output_export_failed"]
        )

    template = json.loads(json.dumps(dict(physics_contract_template)))
    for field in ("task_door_envelope_m", "non_task_door_envelope_m", "support_envelope_m"):
        envelope = template.get(field)
        if (
            not isinstance(envelope, Mapping)
            or _finite_vector(envelope.get("aabb_min"), 3) is None
            or _finite_vector(envelope.get("aabb_max"), 3) is None
        ):
            raise ArticulatedSimReadyReplacementError(
                [f"articulated_authoring_template_envelope_invalid:{field}"]
            )
    physics_contract: dict[str, Any] = {
        key: value
        for key, value in template.items()
        if key
        not in {"task_door_envelope_m", "non_task_door_envelope_m", "support_envelope_m"}
    }
    physics_contract["task_door_link"] = task_door_path
    physics_contract["support_link"] = support_link_path
    physics_contract["required_generated_interior_links"] = [support_link_path]
    envelopes: dict[str, Any] = {support_link_path: template["support_envelope_m"]}
    envelopes[task_door_path] = template["task_door_envelope_m"]
    for door_path in door_link_paths:
        if door_path != task_door_path:
            envelopes[door_path] = template["non_task_door_envelope_m"]
    physics_contract["link_collider_envelopes_m"] = envelopes

    topology_validation = validate_articulated_replacement_topology(
        replacement_usd_path=output, contract=topology_contract
    )
    physics_validation = validate_articulated_replacement_physics(
        replacement_usd_path=output, contract=physics_contract
    )

    receipt: dict[str, Any] = {
        "schema_version": AUTHORING_SCHEMA_VERSION,
        "status": "simready_candidate_statically_admitted",
        "rigged_topology_usd_sha256": _sha256(rigged),
        "output_usd_path": str(output),
        "output_usd_sha256": _sha256(output),
        "task_joint_prim_path": task_joint["path"],
        "support_link_prim_path": support_link_path,
        "task_door_link_prim_path": task_door_path,
        "link_authoring": link_rows,
        "handle": {
            "prim_path": handle_path,
            "source": handle_source,
            "minimum_protrusion_m": minimum_protrusion,
        },
        "generated_interior": {
            "prim_path": f"{support_link_path}/generated_interior",
            "style": "solid_inset_block_v1",
            "inset_m": inset,
        },
        "fixed_base_required_at_insertion": bool(spec.get("fixed_base", True)),
        "authoring_spec": spec,
        "topology_validation": topology_validation,
        "physics_validation": physics_validation,
        "claim_boundary": {
            "authoring_is_deterministic_code_not_vlm_inference": True,
            "native_simulator_qualified": False,
            "physical_equivalence_proven": False,
            "generated_geometry_is_observed_site_truth": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ArticulatedSimReadyReplacementError",
    "AUTHORING_SCHEMA_VERSION",
    "GENERATED_PROVENANCE_VALUE",
    "HANDLE_ROLE_VALUE",
    "OBSERVED_PROVENANCE_VALUE",
    "PHYSICS_SCHEMA_VERSION",
    "PROVENANCE_ATTRIBUTE",
    "TASK_CONTACT_ROLE_ATTRIBUTE",
    "TOPOLOGY_SCHEMA_VERSION",
    "author_articulated_simready_replacement",
    "validate_articulated_replacement_physics",
    "validate_articulated_replacement_topology",
]
