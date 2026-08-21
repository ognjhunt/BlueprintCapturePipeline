"""Derive task-neutral pinch candidates from registered replacement geometry.

This is a deterministic inspection seam, not a grasp synthesizer.  It binds the
task freeze's graph roles to the exact registered USD, selects either the
target-driven moving link or the rigid root task body, and measures a candidate
parallel-jaw pinch.  Native reach, rear-finger access, contact, retention, and
task motion remain explicit downstream gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_task_freeze,
)


SCHEMA_VERSION = "paired_target_interaction_affordance_candidate.v1"
REGISTERED_ASSET_SCHEMA = "registered_replacement_asset.v1"
DEFAULT_PARALLEL_JAW_STROKE_M = 0.085
# Matches native_franka_action_math.GRASP_AXIS_DEGENERACY_TOLERANCE, which is
# where the same collapse is caught on the consuming side.  Pinned equal by
# test so the producer can never seal a pair the author would refuse.
GRIPPER_FRAME_INDEPENDENCE_TOLERANCE = 1.0e-6


class PairedTargetInteractionAffordanceError(ValueError):
    """Stable fail-closed geometry inspection errors."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: str | Path, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedTargetInteractionAffordanceError(code) from exc
    if source.is_symlink() or not isinstance(value, dict):
        raise PairedTargetInteractionAffordanceError(code)
    return source, value


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_source_invalid"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _vector(value: Any, *, length: int, code: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PairedTargetInteractionAffordanceError(code) from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise PairedTargetInteractionAffordanceError(code)
    return result


def _normalize(value: Sequence[float], *, code: str) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise PairedTargetInteractionAffordanceError(code)
    return [float(item) / norm for item in value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right, strict=True)]


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    edge = _subtract(end, start)
    length_squared = _dot(edge, edge)
    if length_squared <= 1.0e-18:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, _dot(_subtract(point, start), edge) / length_squared))
    closest = [start[index] + fraction * edge[index] for index in range(3)]
    return math.dist(point, closest)


def _point_triangle_distance(
    point: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
) -> float:
    edge_a = _subtract(second, first)
    edge_b = _subtract(third, first)
    normal = _cross(edge_a, edge_b)
    normal_norm = math.sqrt(_dot(normal, normal))
    if normal_norm <= 1.0e-12:
        return min(
            _point_segment_distance(point, first, second),
            _point_segment_distance(point, second, third),
            _point_segment_distance(point, third, first),
        )
    unit_normal = [value / normal_norm for value in normal]
    plane_distance = _dot(_subtract(point, first), unit_normal)
    projected = [
        float(point[index]) - plane_distance * unit_normal[index]
        for index in range(3)
    ]
    dot_aa = _dot(edge_a, edge_a)
    dot_ab = _dot(edge_a, edge_b)
    dot_bb = _dot(edge_b, edge_b)
    from_first = _subtract(projected, first)
    dot_pa = _dot(from_first, edge_a)
    dot_pb = _dot(from_first, edge_b)
    denominator = dot_aa * dot_bb - dot_ab * dot_ab
    if denominator > 1.0e-18:
        u = (dot_bb * dot_pa - dot_ab * dot_pb) / denominator
        v = (dot_aa * dot_pb - dot_ab * dot_pa) / denominator
        if u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9:
            return abs(plane_distance)
    return min(
        _point_segment_distance(point, first, second),
        _point_segment_distance(point, second, third),
        _point_segment_distance(point, third, first),
    )


def _mesh_surface_distance(prim: Any, point: Sequence[float]) -> float:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() if mesh else None
    counts = mesh.GetFaceVertexCountsAttr().Get() if mesh else None
    indices = mesh.GetFaceVertexIndicesAttr().Get() if mesh else None
    if not points or not counts or not indices:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_grasp_collision_patch_invalid"
        )
    distances: list[float] = []
    offset = 0
    for count in counts:
        face = [int(value) for value in indices[offset : offset + int(count)]]
        offset += int(count)
        if len(face) < 3 or any(index < 0 or index >= len(points) for index in face):
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_grasp_collision_patch_invalid"
            )
        for index in range(1, len(face) - 1):
            distances.append(
                _point_triangle_distance(
                    point,
                    points[face[0]],
                    points[face[index]],
                    points[face[index + 1]],
                )
            )
    if not distances:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_grasp_collision_patch_invalid"
        )
    return min(distances)


def _custom_vector(prim: Any, key: str, *, code: str) -> list[float]:
    return _vector(prim.GetCustomDataByKey(key), length=3, code=code)


def _bounds(cache: Any, prim: Any, *, local: bool) -> tuple[list[float], list[float]]:
    extent = (
        cache.ComputeRelativeBound(prim, prim) if local else cache.ComputeWorldBound(prim)
    ).ComputeAlignedRange()
    if extent.IsEmpty():
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_link_bounds_empty"
        )
    return (
        [float(item) for item in extent.GetMin()],
        [float(item) for item in extent.GetMax()],
    )


def _corners(low: Sequence[float], high: Sequence[float]) -> list[list[float]]:
    return [
        [low[0] if x else high[0], low[1] if y else high[1], low[2] if z else high[2]]
        for x in (0, 1)
        for y in (0, 1)
        for z in (0, 1)
    ]


def refuse_degenerate_gripper_frame(
    gripper_approach: Sequence[float], pinch_axis: Sequence[float]
) -> None:
    """Refuse an approach/jaw pair that cannot span a parallel-jaw frame.

    Two parallel axes give ``ee_x = ee_y x ee_z = (0, 0, 0)`` and leave the roll
    about the approach unconstrained, so no quaternion exists.  r13-r23 spent
    ten paid runs on the value that collapse produces, because it is a
    well-formed-looking quaternion rather than anything that reads as missing.
    Checked here, against what is about to be sealed, so the refusal names the
    producer rather than surfacing three modules downstream.
    """

    if (
        abs(_dot(gripper_approach, pinch_axis))
        >= 1.0 - GRIPPER_FRAME_INDEPENDENCE_TOLERANCE
    ):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_gripper_frame_axes_degenerate"
        )


def _graph_links(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = freeze.get("articulation_graph")
    if not isinstance(graph, Mapping):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_articulation_graph_missing"
        )
    links = {
        str(row.get("link_id") or ""): dict(row)
        for row in graph.get("links") or []
        if isinstance(row, Mapping)
    }
    joints = {
        str(row.get("joint_id") or ""): dict(row)
        for row in graph.get("joints") or []
        if isinstance(row, Mapping)
    }
    if not links or len(links) != len(graph.get("links") or []):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_graph_links_invalid"
        )
    return links, joints


def materialize_paired_target_interaction_affordance_candidate(
    *,
    task_freeze_path: str | Path,
    registered_asset_receipt_path: str | Path,
    robot_base_position_world_m: Sequence[float],
    output_path: str | Path,
    parallel_jaw_stroke_m: float = DEFAULT_PARALLEL_JAW_STROKE_M,
) -> dict[str, Any]:
    """Measure one graph-bound contact candidate without claiming a grasp."""

    try:
        freeze_path, raw_freeze = _read(
            task_freeze_path, "paired_target_affordance_task_freeze_invalid"
        )
        freeze = validate_task_freeze(raw_freeze)
    except DualTaskRehearsalContractError as exc:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_task_freeze_invalid"
        ) from exc
    registered_path, registered = _read(
        registered_asset_receipt_path,
        "paired_target_affordance_registered_asset_invalid",
    )
    output_usd = registered.get("output_usd")
    if (
        registered.get("schema_version") != REGISTERED_ASSET_SCHEMA
        or registered.get("receipt_digest")
        != canonical_digest(registered, digest_field="receipt_digest")
        or registered.get("task_id") != freeze.get("task_id")
        or registered.get("task_freeze_digest") != freeze.get("task_freeze_digest")
        or not isinstance(output_usd, Mapping)
    ):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_registered_asset_invalid"
        )
    usd_path = Path(str(output_usd.get("path") or "")).expanduser().resolve()
    if (
        usd_path.is_symlink()
        or not usd_path.is_file()
        or usd_path.stat().st_size != output_usd.get("size_bytes")
        or _sha256(usd_path) != output_usd.get("sha256")
    ):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_registered_usd_mismatch"
        )
    base = _vector(
        robot_base_position_world_m,
        length=3,
        code="paired_target_affordance_robot_base_invalid",
    )
    try:
        stroke = float(parallel_jaw_stroke_m)
    except (TypeError, ValueError) as exc:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_gripper_stroke_invalid"
        ) from exc
    if not math.isfinite(stroke) or stroke <= 0.0:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_gripper_stroke_invalid"
        )

    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover - environment guard
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_openusd_runtime_missing"
        ) from exc
    try:
        stage = Usd.Stage.Open(str(usd_path))
    except Exception as exc:  # noqa: BLE001 - pxr exposes versioned Tf errors
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_registered_usd_unreadable"
        ) from exc
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_registered_usd_unreadable"
        )
    root = str(stage.GetDefaultPrim().GetPath())
    links, joints = _graph_links(freeze)
    task_kind = str(freeze.get("task_kind") or "")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    method: str
    target_joint_id: str | None = None
    hinge_world: list[float] | None = None
    hinge_axis: list[float] | None = None

    if task_kind == "articulated_interaction":
        target_ids = sorted(
            joint_id for joint_id, row in joints.items() if row.get("role") == "target"
        )
        if len(target_ids) != 1:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_target_joint_invalid"
            )
        target_joint_id = target_ids[0]
        target = joints[target_joint_id]
        link_id = str(target.get("child_link_id") or "")
        joint_prim = stage.GetPrimAtPath(f"{root}/joints/{target_joint_id}")
        if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_target_joint_missing"
            )
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        body0_targets = list(joint.GetBody0Rel().GetTargets())
        if len(body0_targets) != 1:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_target_joint_binding_invalid"
            )
        body0 = stage.GetPrimAtPath(body0_targets[0])
        local_position = joint.GetLocalPos0Attr().Get()
        local_rotation = joint.GetLocalRot0Attr().Get() or Gf.Quatf(1.0)
        axis_basis = {
            "X": Gf.Vec3d(1, 0, 0),
            "Y": Gf.Vec3d(0, 1, 0),
            "Z": Gf.Vec3d(0, 0, 1),
        }.get(str(joint.GetAxisAttr().Get() or ""))
        if axis_basis is None:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_target_joint_axis_invalid"
            )
        body0_world = UsdGeom.XformCache().GetLocalToWorldTransform(body0)
        hinge_world = [
            float(item) for item in body0_world.Transform(Gf.Vec3d(*local_position))
        ]
        axis_body = Gf.Rotation(local_rotation).TransformDir(axis_basis)
        hinge_axis = _normalize(
            [float(item) for item in body0_world.TransformDir(axis_body)],
            code="paired_target_affordance_target_joint_axis_invalid",
        )
        method = "target_driven_link_far_edge_pinch"
    elif task_kind == "rigid_object_manipulation":
        root_ids = sorted(
            link_id for link_id, row in links.items() if row.get("is_root") is True
        )
        if len(root_ids) != 1:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_rigid_root_link_invalid"
            )
        link_id = root_ids[0]
        method = "rigid_root_thinnest_axis_pinch"
    else:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_task_kind_unsupported"
        )

    link_path = f"{root}/links/{link_id}"
    link = stage.GetPrimAtPath(link_path)
    if not link.IsValid() or not UsdPhysics.RigidBodyAPI(link):
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_link_body_invalid"
        )
    low, high = _bounds(cache, link, local=True)
    center = [(low[index] + high[index]) / 2.0 for index in range(3)]
    spans = [high[index] - low[index] for index in range(3)]
    link_world = UsdGeom.XformCache().GetLocalToWorldTransform(link)
    grasp_patches = [
        prim
        for prim in Usd.PrimRange(link)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and prim.GetCustomDataByKey("blueprint:graspAffordanceRole")
        == "parallel_jaw_outer_rim_patch"
    ]
    if len(grasp_patches) > 1:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_grasp_collision_patch_ambiguous"
        )
    grasp_patch = grasp_patches[0] if grasp_patches else None
    grasp_collision_patch_path = None
    candidate_geometry_modified = False
    if grasp_patch is not None:
        if task_kind != "articulated_interaction":
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_grasp_collision_patch_task_invalid"
            )
        contact_local = _custom_vector(
            grasp_patch,
            "blueprint:graspContactPointLinkM",
            code="paired_target_affordance_grasp_collision_patch_invalid",
        )
        outward_local = _normalize(
            _custom_vector(
                grasp_patch,
                "blueprint:graspApproachOutwardUnitLink",
                code="paired_target_affordance_grasp_collision_patch_invalid",
            ),
            code="paired_target_affordance_grasp_collision_patch_invalid",
        )
        pinch_axis_local = _normalize(
            _custom_vector(
                grasp_patch,
                "blueprint:graspPinchAxisLink",
                code="paired_target_affordance_grasp_collision_patch_invalid",
            ),
            code="paired_target_affordance_grasp_collision_patch_invalid",
        )
        try:
            pinch_span = float(
                grasp_patch.GetCustomDataByKey("blueprint:graspPinchSpanM")
            )
        except (TypeError, ValueError) as exc:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_grasp_collision_patch_invalid"
            ) from exc
        if not math.isfinite(pinch_span) or pinch_span <= 0.0:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_grasp_collision_patch_invalid"
            )
        contact_surface_error_m = _mesh_surface_distance(
            grasp_patch, contact_local
        )
        if contact_surface_error_m > 0.001:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_grasp_contact_off_collider"
            )
        approach = _normalize(
            [
                float(item)
                for item in link_world.TransformDir(Gf.Vec3d(*outward_local))
            ],
            code="paired_target_affordance_grasp_collision_patch_invalid",
        )
        gripper_approach = [-value for value in approach]
        pinch_axis = _normalize(
            [
                float(item)
                for item in link_world.TransformDir(Gf.Vec3d(*pinch_axis_local))
            ],
            code="paired_target_affordance_grasp_collision_patch_invalid",
        )
        contact_world = [
            float(item) for item in link_world.Transform(Gf.Vec3d(*contact_local))
        ]
        method = "source_derived_grasp_collision_patch"
        gripper_approach_source = "grasp_patch_front_normal_inward"
        standoff_axis_source = "grasp_patch_front_normal_outward"
        grasp_collision_patch_path = str(grasp_patch.GetPath())
        candidate_geometry_modified = True
    elif task_kind == "rigid_object_manipulation":
        pinch_axis_index = min(range(3), key=spans.__getitem__)
        pinch_axis_local = [
            1.0 if index == pinch_axis_index else 0.0 for index in range(3)
        ]
        pinch_axis = _normalize(
            [float(item) for item in link_world.TransformDir(Gf.Vec3d(*pinch_axis_local))],
            code="paired_target_affordance_pinch_axis_invalid",
        )
        contact_local = center
        contact_world = [
            float(item) for item in link_world.Transform(Gf.Vec3d(*contact_local))
        ]
        approach = _normalize(
            [contact_world[index] - base[index] for index in range(3)],
            code="paired_target_affordance_approach_invalid",
        )
        # A free body has no privileged in-plane direction, so the standoff
        # direction is also the direction the gripper travels in along.  These
        # are two different quantities that happen to coincide here; the
        # articulated branch below is where they must not.
        gripper_approach = list(approach)
        gripper_approach_source = "base_to_contact_direction"
        standoff_axis_source = "base_to_contact_direction"
    else:
        assert hinge_world is not None and hinge_axis is not None
        inverse = link_world.GetInverse()
        hinge_local = [
            float(item) for item in inverse.Transform(Gf.Vec3d(*hinge_world))
        ]
        hinge_axis_local = _normalize(
            [float(item) for item in inverse.TransformDir(Gf.Vec3d(*hinge_axis))],
            code="paired_target_affordance_target_joint_axis_invalid",
        )
        perpendicular_axes = [
            index for index in range(3) if abs(hinge_axis_local[index]) <= 1.0e-6
        ]
        if not perpendicular_axes:
            raise PairedTargetInteractionAffordanceError(
                "paired_target_affordance_panel_basis_invalid"
            )
        normal_axis_index = min(perpendicular_axes, key=spans.__getitem__)
        normal = [1.0 if index == normal_axis_index else 0.0 for index in range(3)]
        offset = [center[index] - hinge_local[index] for index in range(3)]
        radial = _normalize(
            _cross(hinge_axis_local, normal),
            code="paired_target_affordance_radial_direction_invalid",
        )
        if _dot(offset, radial) < 0.0:
            radial = [-item for item in radial]
        normal_world = _normalize(
            [float(item) for item in link_world.TransformDir(Gf.Vec3d(*normal))],
            code="paired_target_affordance_panel_normal_invalid",
        )
        center_world = [
            float(item) for item in link_world.Transform(Gf.Vec3d(*center))
        ]
        if _dot([base[index] - center_world[index] for index in range(3)], normal_world) < 0.0:
            normal = [-item for item in normal]
            normal_world = [-item for item in normal_world]
        corners = _corners(low, high)
        far_radial = max(_dot(point, radial) for point in corners)
        middle_axis = _dot(center, hinge_axis_local)
        near_normal = max(_dot(point, normal) for point in corners)
        contact_local = [
            radial[index] * far_radial
            + hinge_axis_local[index] * middle_axis
            + normal[index] * near_normal
            for index in range(3)
        ]
        contact_world = [
            float(item) for item in link_world.Transform(Gf.Vec3d(*contact_local))
        ]
        pinch_axis_local = normal
        pinch_axis = normal_world
        # A parallel-jaw gripper cannot acquire a panel by translating along
        # the axis its jaws close on: the leading finger/knuckle reaches the
        # panel before the panel can enter between the open fingers.  Stage the
        # gripper outside the free edge instead, then travel hinge-ward along
        # the in-plane radial.  Arena controls c3/c4 measured both sides of the
        # old defect: exact normal-axis targeting struck the door with a
        # non-fingertip body, while a 25 mm normal standoff produced no contact.
        #
        # ``radial`` points from the hinge toward the selected free edge, so it
        # is the outward standoff direction.  The tool approach axis is its
        # opposite and remains independent of the jaw axis.
        radial_world = _normalize(
            [float(item) for item in link_world.TransformDir(Gf.Vec3d(*radial))],
            code="paired_target_affordance_radial_direction_invalid",
        )
        approach = radial_world

        # ``radial`` already points from the hinge toward the panel centre, and
        # ``contact_local`` is taken at ``far_radial`` -- the free edge.  The
        # gripper therefore travels INWARD along the panel plane, from beyond
        # the free edge toward the hinge, which is the one in-plane direction
        # that leaves the palm and the rear finger in free space rather than
        # inside the appliance.  Hinge-ward is the only sign that is not a
        # wrist buried in the cabinet, so the sign is not free either.
        gripper_approach = [-value for value in radial_world]
        gripper_approach_source = "panel_plane_radial_inward_from_free_edge"
        standoff_axis_source = "panel_plane_radial_outward_from_free_edge"
    if grasp_patch is None:
        pinch_span = sum(
            abs(pinch_axis_local[index]) * spans[index] for index in range(3)
        )
    refuse_degenerate_gripper_frame(gripper_approach, pinch_axis)
    measured_collision_paths = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(link)
        if prim.IsA(UsdGeom.Boundable) and bool(UsdPhysics.CollisionAPI(prim))
    )
    if not measured_collision_paths:
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_collision_region_missing"
        )

    blockers = [
        "native_reach_and_two_finger_contact_unproven",
        "native_grasp_retention_unproven",
        "native_support_and_rear_finger_clearance_unproven",
    ]
    if pinch_span > stroke:
        blockers.append("candidate_pinch_span_exceeds_parallel_jaw_stroke")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_geometry_materialized_requires_native_contact",
        "scene_id": registered["scene_id"],
        "task_id": freeze["task_id"],
        "task_kind": task_kind,
        "asset_id": registered["asset_id"],
        "task_freeze": _record(
            freeze_path, task_freeze_digest=freeze["task_freeze_digest"]
        ),
        "registered_asset": _record(
            registered_path, receipt_digest=registered["receipt_digest"]
        ),
        "registered_usd": _record(usd_path),
        "robot_base_position_world_m": base,
        "selection_contract": {
            "method": method,
            "gripper_approach_axis_source": gripper_approach_source,
            "standoff_axis_source": standoff_axis_source,
            "link_selected_from_articulation_graph_role": True,
            "object_label_or_task_id_geometry_shortcut_used": False,
            "candidate_geometry_authored_or_modified": candidate_geometry_modified,
        },
        "candidate": {
            "link_id": link_id,
            "link_prim_path": link_path,
            "target_joint_id": target_joint_id,
            # Contact sensors attach to rigid bodies, not collision-mesh
            # children.  Preserve the measured collision region separately.
            "contact_body_prim_paths": [link_path],
            "measured_collision_prim_paths": measured_collision_paths,
            "grasp_collision_patch_prim_path": grasp_collision_patch_path,
            "contact_point_to_grasp_collider_surface_m": (
                contact_surface_error_m if grasp_patch is not None else None
            ),
            "contact_point_link_m": contact_local,
            "contact_point_registered_stage_m": contact_world,
            "approach_unit_registered_stage": approach,
            # Consumed as an outward standoff translation.  For a hinged panel
            # it is the opposite of the tool approach below, and both are
            # perpendicular to the jaw axis so the open fingers can surround
            # the free edge before closing across the panel thickness.
            "gripper_approach_axis_registered_stage": gripper_approach,
            "pinch_axis_registered_stage": pinch_axis,
            "pinch_span_m": pinch_span,
            "parallel_jaw_stroke_m": stroke,
            "pinch_span_within_stroke": pinch_span <= stroke,
            "link_aabb_link_local_m": {"minimum": low, "maximum": high},
            "hinge_point_registered_stage_m": hinge_world,
            "hinge_axis_registered_stage_unit": hinge_axis,
        },
        "native_contact_execution_authorized": False,
        "native_contact_executed": False,
        "blockers": sorted(blockers),
        "generated_output_is_capture_or_physical_evidence": False,
        "claim_boundary": (
            "deterministic_registered_usd_geometry_candidate_only;not_ik_reach_"
            "contact_grasp_retention_task_success_policy_or_physical_evidence"
        ),
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise PairedTargetInteractionAffordanceError(
            "paired_target_affordance_destination_exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json.loads(json.dumps(result))


__all__ = [
    "DEFAULT_PARALLEL_JAW_STROKE_M",
    "GRIPPER_FRAME_INDEPENDENCE_TOLERANCE",
    "PairedTargetInteractionAffordanceError",
    "SCHEMA_VERSION",
    "materialize_paired_target_interaction_affordance_candidate",
    "refuse_degenerate_gripper_frame",
]
