"""Deterministic articulated-member sweep clearance against bound obstacles.

This is an early, simulator-free rejection gate.  It rotates an observed
candidate member centerline through a preregistered angle and tests it against
exact source-obstacle AABBs.  A collision of a zero-thickness centerline is a
strong rejection: adding the real member thickness cannot restore clearance.
Passing remains candidate evidence and never replaces native collider, IK, or
contact qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "articulated_workspace_clearance.v1"
SAGE_OBSTACLE_INVENTORY_SCHEMA_VERSION = "sage_sweep_obstacle_inventory.v1"
SAGE_MESH_SWEEP_SCHEMA_VERSION = "articulated_sage_mesh_sweep.v1"


class ArticulatedWorkspaceClearanceError(ValueError):
    """Stable, sorted sweep-clearance failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_vector(value: Any, length: int, error: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ArticulatedWorkspaceClearanceError([error])
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError([error]) from exc
    if any(not math.isfinite(item) for item in result):
        raise ArticulatedWorkspaceClearanceError([error])
    return result


def _segment_aabb_intersection(
    start: Sequence[float], end: Sequence[float], minimum: Sequence[float], maximum: Sequence[float]
) -> tuple[bool, list[float] | None]:
    """Liang-Barsky segment/AABB intersection in the horizontal plane."""

    low = 0.0
    high = 1.0
    for axis in range(2):
        delta = float(end[axis]) - float(start[axis])
        if abs(delta) <= 1e-15:
            if float(start[axis]) < float(minimum[axis]) or float(start[axis]) > float(maximum[axis]):
                return False, None
            continue
        axis_low = (float(minimum[axis]) - float(start[axis])) / delta
        axis_high = (float(maximum[axis]) - float(start[axis])) / delta
        if axis_low > axis_high:
            axis_low, axis_high = axis_high, axis_low
        low = max(low, axis_low)
        high = min(high, axis_high)
        if low > high:
            return False, None
    return True, [
        float(start[axis]) + low * (float(end[axis]) - float(start[axis]))
        for axis in range(2)
    ]


def evaluate_revolute_member_sweep(
    *,
    hinge_origin_world_m: Sequence[float],
    closed_endpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    start_angle_degrees: float,
    end_angle_degrees: float,
    obstacles: Sequence[Mapping[str, Any]],
    angular_resolution_degrees: float = 0.25,
    member_half_thickness_m: float = 0.0,
) -> dict[str, Any]:
    """Return the first deterministic centerline collision, if one exists."""

    hinge = _finite_vector(hinge_origin_world_m, 3, "sweep_hinge_invalid")
    endpoint = _finite_vector(closed_endpoint_world_m, 3, "sweep_endpoint_invalid")
    vertical = _finite_vector(member_vertical_interval_m, 2, "sweep_vertical_interval_invalid")
    if vertical[0] >= vertical[1]:
        raise ArticulatedWorkspaceClearanceError(["sweep_vertical_interval_invalid"])
    try:
        start_angle = float(start_angle_degrees)
        end_angle = float(end_angle_degrees)
        resolution = float(angular_resolution_degrees)
        half_thickness = float(member_half_thickness_m)
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"]) from exc
    if (
        not all(math.isfinite(value) for value in (start_angle, end_angle, resolution, half_thickness))
        or end_angle == start_angle
        or resolution <= 0.0
        or half_thickness < 0.0
    ):
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"])
    if abs(endpoint[2] - hinge[2]) > 1e-9:
        raise ArticulatedWorkspaceClearanceError(["sweep_centerline_not_horizontal"])
    radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
    if radius <= 0.0:
        raise ArticulatedWorkspaceClearanceError(["sweep_member_radius_invalid"])
    source_angle = math.atan2(endpoint[1] - hinge[1], endpoint[0] - hinge[0])

    normalized_obstacles: list[dict[str, Any]] = []
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, Mapping):
            raise ArticulatedWorkspaceClearanceError([f"sweep_obstacle_{index}_invalid"])
        minimum = _finite_vector(
            obstacle.get("world_aabb_min_m"), 3, f"sweep_obstacle_{index}_aabb_invalid"
        )
        maximum = _finite_vector(
            obstacle.get("world_aabb_max_m"), 3, f"sweep_obstacle_{index}_aabb_invalid"
        )
        if any(minimum[axis] >= maximum[axis] for axis in range(3)):
            raise ArticulatedWorkspaceClearanceError([f"sweep_obstacle_{index}_aabb_invalid"])
        normalized_obstacles.append(
            {
                "obstacle_id": str(obstacle.get("obstacle_id") or f"obstacle_{index}"),
                "world_aabb_min_m": minimum,
                "world_aabb_max_m": maximum,
                "source_receipt_digest": obstacle.get("source_receipt_digest"),
            }
        )

    direction = 1.0 if end_angle > start_angle else -1.0
    step_count = int(math.ceil(abs(end_angle - start_angle) / resolution))
    angles = [
        start_angle + index * direction * resolution for index in range(step_count)
    ] + [end_angle]
    first_collision: dict[str, Any] | None = None
    collision_obstacle_ids: set[str] = set()
    for angle_degrees in angles:
        angle = source_angle + math.radians(angle_degrees)
        rotated_endpoint = [
            hinge[0] + radius * math.cos(angle),
            hinge[1] + radius * math.sin(angle),
        ]
        for obstacle in normalized_obstacles:
            minimum = obstacle["world_aabb_min_m"]
            maximum = obstacle["world_aabb_max_m"]
            vertical_overlap = min(vertical[1], maximum[2]) > max(vertical[0], minimum[2])
            if not vertical_overlap:
                continue
            inflated_minimum = [minimum[0] - half_thickness, minimum[1] - half_thickness]
            inflated_maximum = [maximum[0] + half_thickness, maximum[1] + half_thickness]
            intersects, point = _segment_aabb_intersection(
                hinge, rotated_endpoint, inflated_minimum, inflated_maximum
            )
            if not intersects:
                continue
            collision_obstacle_ids.add(obstacle["obstacle_id"])
            if first_collision is None:
                first_collision = {
                    "angle_degrees": round(angle_degrees, 9),
                    "obstacle_id": obstacle["obstacle_id"],
                    "intersection_xy_world_m": [round(value, 9) for value in point or []],
                    "rotated_endpoint_xy_world_m": [
                        round(value, 9) for value in rotated_endpoint
                    ],
                }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_by_observed_obstacle" if first_collision else "clearance_candidate_only",
        "sweep": {
            "hinge_origin_world_m": hinge,
            "closed_endpoint_world_m": endpoint,
            "member_radius_m": radius,
            "member_vertical_interval_m": vertical,
            "start_angle_degrees": start_angle,
            "end_angle_degrees": end_angle,
            "angular_resolution_degrees": resolution,
            "member_half_thickness_m": half_thickness,
            "sample_count": len(angles),
        },
        "obstacles": normalized_obstacles,
        "first_collision": first_collision,
        "collision_obstacle_ids": sorted(collision_obstacle_ids),
        "claim_boundary": {
            "zero_thickness_centerline_collision_is_strong_rejection": half_thickness == 0.0,
            "clear_result_is_not_native_dynamic_qualification": True,
            "franka_base_pose_resolved": False,
            "ik_or_contact_qualified": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def load_bound_collision_obstacle(path: str | Path) -> dict[str, Any]:
    """Load one exact SAGE identity receipt as a sweep obstacle."""

    source = Path(path).expanduser().resolve()
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_invalid"]) from exc
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != "interiorgs_sage_collision_identity.v1":
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_invalid"])
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_receipt_digest_invalid"])
    matches = receipt.get("whole_object_matches")
    if not isinstance(matches, list) or len(matches) != 1:
        raise ArticulatedWorkspaceClearanceError(["sweep_obstacle_unique_collision_match_missing"])
    match = matches[0]
    target = receipt.get("target") or {}
    return {
        "obstacle_id": f"{target.get('semantic_label')}:{target.get('interiorgs_instance_id')}",
        "world_aabb_min_m": match["world_aabb_min_m"],
        "world_aabb_max_m": match["world_aabb_max_m"],
        "source_receipt_digest": receipt["receipt_digest"],
        "source_file_sha256": _sha256(source),
    }


def inventory_sage_sweep_obstacles(
    *,
    sage_collision_usd_path: str | Path,
    target_collision_identity_receipt_path: str | Path,
    hinge_origin_world_m: Sequence[float],
    closed_endpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    broadphase_padding_m: float = 0.05,
) -> dict[str, Any]:
    """Survey every SAGE collision mesh intersecting a member-sweep broadphase.

    The broadphase is the complete horizontal disk swept by the member radius,
    conservatively represented as an AABB.  Only the uniquely identity-bound
    target mesh is excluded.  The resulting obstacle list is exhaustive for
    the source stage and broadphase, but its AABBs remain an early rejection
    approximation rather than native dynamic-contact proof.
    """

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_openusd_runtime_missing"]
        ) from exc

    collision = Path(sage_collision_usd_path).expanduser().resolve()
    identity_path = Path(target_collision_identity_receipt_path).expanduser().resolve()
    if not collision.is_file() or collision.is_symlink():
        raise ArticulatedWorkspaceClearanceError(["sage_sweep_collision_usd_missing"])
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_target_identity_invalid"]
        ) from exc
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema_version") != "interiorgs_sage_collision_identity.v1"
        or identity.get("receipt_digest")
        != canonical_digest(identity, digest_field="receipt_digest")
    ):
        raise ArticulatedWorkspaceClearanceError(["sage_sweep_target_identity_invalid"])
    whole_matches = identity.get("whole_object_matches")
    if not isinstance(whole_matches, list) or len(whole_matches) != 1:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_target_unique_collision_match_missing"]
        )
    collision_digest = _sha256(collision)
    if (
        (identity.get("source_files") or {})
        .get("sage_collision_usd", {})
        .get("sha256")
        != collision_digest
    ):
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_collision_source_digest_mismatch"]
        )

    hinge = _finite_vector(hinge_origin_world_m, 3, "sweep_hinge_invalid")
    endpoint = _finite_vector(closed_endpoint_world_m, 3, "sweep_endpoint_invalid")
    vertical = _finite_vector(
        member_vertical_interval_m, 2, "sweep_vertical_interval_invalid"
    )
    if vertical[0] >= vertical[1]:
        raise ArticulatedWorkspaceClearanceError(["sweep_vertical_interval_invalid"])
    try:
        padding = float(broadphase_padding_m)
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_broadphase_padding_invalid"]
        ) from exc
    if not math.isfinite(padding) or padding < 0.0:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_sweep_broadphase_padding_invalid"]
        )
    radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
    if radius <= 0.0:
        raise ArticulatedWorkspaceClearanceError(["sweep_member_radius_invalid"])
    broadphase_min = [
        hinge[0] - radius - padding,
        hinge[1] - radius - padding,
        vertical[0] - padding,
    ]
    broadphase_max = [
        hinge[0] + radius + padding,
        hinge[1] + radius + padding,
        vertical[1] + padding,
    ]

    stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedWorkspaceClearanceError(["sage_sweep_collision_usd_open_failed"])
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ArticulatedWorkspaceClearanceError(["sage_sweep_stage_not_z_up"])
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise ArticulatedWorkspaceClearanceError(["sage_sweep_stage_not_meter_units"])
    excluded_paths = sorted(str(row["prim_path"]) for row in whole_matches)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    traversed_mesh_count = 0
    collision_mesh_count = 0
    obstacles: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default()) or []
        if not points or not counts:
            continue
        traversed_mesh_count += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_mesh_count += 1
        prim_path = str(prim.GetPath())
        if prim_path in excluded_paths:
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        raw_minimum = aligned.GetMin()
        raw_maximum = aligned.GetMax()
        minimum = [float(raw_minimum[index]) for index in range(3)]
        maximum = [float(raw_maximum[index]) for index in range(3)]
        intersects = all(
            min(maximum[axis], broadphase_max[axis])
            > max(minimum[axis], broadphase_min[axis])
            for axis in range(3)
        )
        if not intersects:
            continue
        obstacles.append(
            {
                "obstacle_id": f"sage_prim:{prim_path}",
                "source_prim_path": prim_path,
                "world_aabb_min_m": [round(value, 9) for value in minimum],
                "world_aabb_max_m": [round(value, 9) for value in maximum],
                "point_count": len(points),
                "face_count": len(counts),
            }
        )
    obstacles.sort(key=lambda row: row["source_prim_path"])
    result: dict[str, Any] = {
        "schema_version": SAGE_OBSTACLE_INVENTORY_SCHEMA_VERSION,
        "status": "completed",
        "source": {
            "sage_collision_usd_path": collision.name,
            "sage_collision_usd_sha256": collision_digest,
            "target_collision_identity_receipt_sha256": _sha256(identity_path),
            "target_collision_identity_receipt_digest": identity["receipt_digest"],
        },
        "coordinate_frame": {
            "up_axis": "Z",
            "meters_per_unit": meters_per_unit,
            "transform_applied": "identity",
        },
        "broadphase": {
            "hinge_origin_world_m": hinge,
            "closed_endpoint_world_m": endpoint,
            "member_radius_m": radius,
            "member_vertical_interval_m": vertical,
            "padding_m": padding,
            "world_aabb_min_m": broadphase_min,
            "world_aabb_max_m": broadphase_max,
        },
        "excluded_target_prim_paths": excluded_paths,
        "traversed_mesh_count": traversed_mesh_count,
        "collision_mesh_count": collision_mesh_count,
        "obstacles": obstacles,
        "obstacle_count": len(obstacles),
        "claim_boundary": {
            "full_sage_stage_surveyed": True,
            "only_bound_target_collision_excluded": True,
            "obstacle_aabbs_are_conservative": True,
            "clear_inventory_is_not_native_dynamic_qualification": True,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def obstacles_from_sage_sweep_inventory(
    value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate an inventory and return sweep-compatible, digest-bound rows."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SAGE_OBSTACLE_INVENTORY_SCHEMA_VERSION:
        errors.append("sage_sweep_inventory_schema_invalid")
    if payload.get("status") != "completed":
        errors.append("sage_sweep_inventory_status_invalid")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("sage_sweep_inventory_digest_invalid")
    obstacles = payload.get("obstacles")
    if not isinstance(obstacles, list) or payload.get("obstacle_count") != len(obstacles):
        errors.append("sage_sweep_inventory_obstacles_invalid")
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)
    return [
        {
            "obstacle_id": row["obstacle_id"],
            "world_aabb_min_m": row["world_aabb_min_m"],
            "world_aabb_max_m": row["world_aabb_max_m"],
            "source_receipt_digest": payload["receipt_digest"],
        }
        for row in obstacles
    ]


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _triangle_intersects_axis_aligned_box(
    triangle: Sequence[Sequence[float]], half_extents: Sequence[float]
) -> bool:
    """Exact convex SAT test for one triangle and a box centered at the origin."""

    edges = [
        [triangle[1][axis] - triangle[0][axis] for axis in range(3)],
        [triangle[2][axis] - triangle[1][axis] for axis in range(3)],
        [triangle[0][axis] - triangle[2][axis] for axis in range(3)],
    ]
    box_axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    axes: list[Sequence[float]] = [*box_axes, _cross(edges[0], edges[1])]
    axes.extend(_cross(edge, axis) for edge in edges for axis in box_axes)
    for axis in axes:
        norm_sq = _dot(axis, axis)
        if norm_sq <= 1e-24:
            continue
        projections = [_dot(vertex, axis) for vertex in triangle]
        radius = sum(abs(float(axis[index])) * float(half_extents[index]) for index in range(3))
        if min(projections) > radius + 1e-12 or max(projections) < -radius - 1e-12:
            return False
    return True


def evaluate_revolute_member_sweep_against_sage_meshes(
    *,
    sage_collision_usd_path: str | Path,
    obstacle_inventory: Mapping[str, Any],
    hinge_origin_world_m: Sequence[float],
    closed_endpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    start_angle_degrees: float,
    end_angle_degrees: float,
    member_half_thickness_m: float,
    angular_resolution_degrees: float = 0.25,
) -> dict[str, Any]:
    """Narrow an exhaustive SAGE AABB inventory using exact mesh triangles.

    At each preregistered angle the moving member is a rectangular prism.  The
    test transforms every inventoried SAGE triangle into that prism's frame and
    applies the complete triangle/box separating-axis test.  This removes AABB
    false positives while retaining the full-stage inventory as the broadphase.
    """

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_mesh_sweep_openusd_runtime_missing"]
        ) from exc
    inventory = json.loads(json.dumps(obstacle_inventory))
    obstacles_from_sage_sweep_inventory(inventory)
    collision = Path(sage_collision_usd_path).expanduser().resolve()
    if not collision.is_file() or collision.is_symlink():
        raise ArticulatedWorkspaceClearanceError(["sage_mesh_sweep_collision_usd_missing"])
    if inventory["source"]["sage_collision_usd_sha256"] != _sha256(collision):
        raise ArticulatedWorkspaceClearanceError(
            ["sage_mesh_sweep_collision_source_digest_mismatch"]
        )
    hinge = _finite_vector(hinge_origin_world_m, 3, "sweep_hinge_invalid")
    endpoint = _finite_vector(closed_endpoint_world_m, 3, "sweep_endpoint_invalid")
    vertical = _finite_vector(
        member_vertical_interval_m, 2, "sweep_vertical_interval_invalid"
    )
    try:
        start_angle = float(start_angle_degrees)
        end_angle = float(end_angle_degrees)
        resolution = float(angular_resolution_degrees)
        half_thickness = float(member_half_thickness_m)
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"]) from exc
    if (
        vertical[0] >= vertical[1]
        or not all(
            math.isfinite(value)
            for value in (start_angle, end_angle, resolution, half_thickness)
        )
        or end_angle == start_angle
        or resolution <= 0.0
        or half_thickness <= 0.0
    ):
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"])
    radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
    if radius <= 0.0:
        raise ArticulatedWorkspaceClearanceError(["sweep_member_radius_invalid"])
    source_angle = math.atan2(endpoint[1] - hinge[1], endpoint[0] - hinge[0])
    direction = 1.0 if end_angle > start_angle else -1.0
    step_count = int(math.ceil(abs(end_angle - start_angle) / resolution))
    angles = [
        start_angle + index * direction * resolution for index in range(step_count)
    ] + [end_angle]

    stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedWorkspaceClearanceError(["sage_mesh_sweep_collision_usd_open_failed"])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_rows: list[dict[str, Any]] = []
    mesh_triangles: list[
        tuple[str, list[tuple[list[list[float]], list[float], list[float]]]]
    ] = []
    for row in inventory["obstacles"]:
        prim_path = str(row["source_prim_path"])
        prim = stage.GetPrimAtPath(prim_path)
        if (
            not prim.IsValid()
            or not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            raise ArticulatedWorkspaceClearanceError(
                [f"sage_mesh_sweep_obstacle_prim_invalid:{prim_path}"]
            )
        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        transform = xform_cache.GetLocalToWorldTransform(prim)
        world_points = [
            [float(value) for value in transform.Transform(point)] for point in local_points
        ]
        triangles: list[list[list[float]]] = []
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            for offset in range(1, count - 1):
                triangles.append(
                    [world_points[face[0]], world_points[face[offset]], world_points[face[offset + 1]]]
                )
        geometry_digest = canonical_digest(
            {"world_points": world_points, "triangles": triangles}
        )
        mesh_rows.append(
            {
                "source_prim_path": prim_path,
                "world_point_count": len(world_points),
                "triangle_count": len(triangles),
                "world_triangle_geometry_digest": geometry_digest,
            }
        )
        mesh_triangles.append(
            (
                prim_path,
                [
                    (
                        triangle,
                        [min(point[axis] for point in triangle) for axis in range(3)],
                        [max(point[axis] for point in triangle) for axis in range(3)],
                    )
                    for triangle in triangles
                ],
            )
        )

    z_center = 0.5 * (vertical[0] + vertical[1])
    half_extents = [0.5 * radius, half_thickness, 0.5 * (vertical[1] - vertical[0])]
    first_collision: dict[str, Any] | None = None
    collision_prim_paths: set[str] = set()
    tested_triangle_poses = 0
    broadphase_rejected_triangle_poses = 0
    for angle_degrees in angles:
        angle = source_angle + math.radians(angle_degrees)
        radial = [math.cos(angle), math.sin(angle), 0.0]
        tangent = [-math.sin(angle), math.cos(angle), 0.0]
        center = [
            hinge[0] + 0.5 * radius * radial[0],
            hinge[1] + 0.5 * radius * radial[1],
            z_center,
        ]
        door_world_half_extents = [
            abs(radial[0]) * half_extents[0] + abs(tangent[0]) * half_extents[1],
            abs(radial[1]) * half_extents[0] + abs(tangent[1]) * half_extents[1],
            half_extents[2],
        ]
        door_minimum = [
            center[axis] - door_world_half_extents[axis] for axis in range(3)
        ]
        door_maximum = [
            center[axis] + door_world_half_extents[axis] for axis in range(3)
        ]
        for prim_path, triangles in mesh_triangles:
            for triangle_index, (triangle, triangle_minimum, triangle_maximum) in enumerate(
                triangles
            ):
                if any(
                    min(door_maximum[axis], triangle_maximum[axis])
                    < max(door_minimum[axis], triangle_minimum[axis])
                    for axis in range(3)
                ):
                    broadphase_rejected_triangle_poses += 1
                    continue
                local_triangle = []
                for point in triangle:
                    relative = [point[index] - center[index] for index in range(3)]
                    local_triangle.append(
                        [_dot(relative, radial), _dot(relative, tangent), relative[2]]
                    )
                tested_triangle_poses += 1
                if not _triangle_intersects_axis_aligned_box(local_triangle, half_extents):
                    continue
                collision_prim_paths.add(prim_path)
                if first_collision is None:
                    first_collision = {
                        "angle_degrees": round(angle_degrees, 9),
                        "source_prim_path": prim_path,
                        "triangle_index": triangle_index,
                    }
                break

    result: dict[str, Any] = {
        "schema_version": SAGE_MESH_SWEEP_SCHEMA_VERSION,
        "status": (
            "blocked_by_exact_sage_mesh_contact"
            if first_collision
            else "exact_sage_mesh_clearance_candidate_only"
        ),
        "source": {
            "sage_collision_usd_sha256": _sha256(collision),
            "obstacle_inventory_receipt_digest": inventory["receipt_digest"],
        },
        "sweep": {
            "hinge_origin_world_m": hinge,
            "closed_endpoint_world_m": endpoint,
            "member_vertical_interval_m": vertical,
            "member_radius_m": radius,
            "member_half_thickness_m": half_thickness,
            "start_angle_degrees": start_angle,
            "end_angle_degrees": end_angle,
            "angular_resolution_degrees": resolution,
            "sample_count": len(angles),
        },
        "mesh_geometry": mesh_rows,
        "first_collision": first_collision,
        "collision_prim_paths": sorted(collision_prim_paths),
        "tested_triangle_poses": tested_triangle_poses,
        "broadphase_rejected_triangle_poses": broadphase_rejected_triangle_poses,
        "claim_boundary": {
            "triangle_prism_intersection_tested": True,
            "full_stage_inventory_is_bound_broadphase": True,
            "clear_result_is_not_native_dynamic_qualification": True,
            "franka_base_pose_resolved": False,
            "ik_or_contact_qualified": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


DOOR_STATE_CLEARANCE_SCHEMA_VERSION = "articulated_door_state_clearance.v1"
DOOR_STATE_MINIMUM_COUNT = 12
_STATIC_OBSTACLE_CLASSES = ("replacement_body", "replacement_lower_door", "franka_base")


def _axis_aligned_box_triangles(
    minimum: Sequence[float], maximum: Sequence[float]
) -> list[list[list[float]]]:
    corners = [
        [minimum[0] if sx == 0 else maximum[0],
         minimum[1] if sy == 0 else maximum[1],
         minimum[2] if sz == 0 else maximum[2]]
        for sx in (0, 1)
        for sy in (0, 1)
        for sz in (0, 1)
    ]
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles: list[list[list[float]]] = []
    for quad in quads:
        triangles.append([corners[quad[0]], corners[quad[1]], corners[quad[2]]])
        triangles.append([corners[quad[0]], corners[quad[2]], corners[quad[3]]])
    return triangles


def evaluate_frozen_door_state_clearance(
    *,
    sage_collision_usd_path: str | Path,
    obstacle_inventory: Mapping[str, Any],
    hinge_origin_world_m: Sequence[float],
    closed_endpoint_world_m: Sequence[float],
    member_vertical_interval_m: Sequence[float],
    member_half_thickness_m: float,
    door_state_angles_degrees: Sequence[float],
    required_maximum_angle_degrees: float,
    static_box_obstacles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Test the frozen discrete door states against SAGE and bound static boxes.

    The continuous exact sweep already narrows AABB false positives; this
    matrix pins the preregistered door states themselves (closed through the
    admitted open range) and additionally accepts labeled static boxes for the
    authored replacement body, the locked lower door, and the resolved Franka
    base. Classes that are not yet bound stay explicit in the claim boundary so
    scenario admission can fail closed on missing self-geometry rather than
    silently treating an unbound class as clear.
    """

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_mesh_sweep_openusd_runtime_missing"]
        ) from exc

    errors: list[str] = []
    states: list[float] = []
    for value in door_state_angles_degrees:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append("door_state_angles_invalid")
            break
        states.append(float(value))
    if len(states) < DOOR_STATE_MINIMUM_COUNT:
        errors.append(
            f"door_state_angles_below_minimum_count:{len(states)}<{DOOR_STATE_MINIMUM_COUNT}"
        )
    if any(late <= early for early, late in zip(states, states[1:])):
        errors.append("door_state_angles_not_strictly_increasing")
    if states and states[0] != 0.0:
        errors.append("door_state_angles_must_start_closed")
    try:
        required_maximum = float(required_maximum_angle_degrees)
    except (TypeError, ValueError):
        required_maximum = math.inf
        errors.append("door_state_required_maximum_invalid")
    if states and states[-1] < required_maximum:
        errors.append(
            "door_state_angles_do_not_reach_required_maximum:"
            f"{states[-1]!r}<{required_maximum!r}"
        )

    boxes: list[dict[str, Any]] = []
    for index, row in enumerate(static_box_obstacles):
        if not isinstance(row, Mapping):
            errors.append(f"door_state_static_box_invalid:{index}")
            continue
        label = str(row.get("label") or "")
        obstacle_class = str(row.get("obstacle_class") or "")
        minimum = _finite_vector(
            row.get("aabb_min"), 3, f"door_state_static_box_invalid:{index}"
        )
        maximum = _finite_vector(
            row.get("aabb_max"), 3, f"door_state_static_box_invalid:{index}"
        )
        if (
            not label
            or obstacle_class not in _STATIC_OBSTACLE_CLASSES
            or any(minimum[axis] >= maximum[axis] for axis in range(3))
        ):
            errors.append(f"door_state_static_box_invalid:{index}")
            continue
        boxes.append(
            {
                "label": label,
                "obstacle_class": obstacle_class,
                "aabb_min": minimum,
                "aabb_max": maximum,
                "triangles": [
                    (
                        triangle,
                        [min(point[axis] for point in triangle) for axis in range(3)],
                        [max(point[axis] for point in triangle) for axis in range(3)],
                    )
                    for triangle in _axis_aligned_box_triangles(minimum, maximum)
                ],
            }
        )
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)

    inventory = json.loads(json.dumps(obstacle_inventory))
    obstacles_from_sage_sweep_inventory(inventory)
    collision = Path(sage_collision_usd_path).expanduser().resolve()
    if not collision.is_file() or collision.is_symlink():
        raise ArticulatedWorkspaceClearanceError(["sage_mesh_sweep_collision_usd_missing"])
    if inventory["source"]["sage_collision_usd_sha256"] != _sha256(collision):
        raise ArticulatedWorkspaceClearanceError(
            ["sage_mesh_sweep_collision_source_digest_mismatch"]
        )
    hinge = _finite_vector(hinge_origin_world_m, 3, "sweep_hinge_invalid")
    endpoint = _finite_vector(closed_endpoint_world_m, 3, "sweep_endpoint_invalid")
    vertical = _finite_vector(
        member_vertical_interval_m, 2, "sweep_vertical_interval_invalid"
    )
    try:
        half_thickness = float(member_half_thickness_m)
    except (TypeError, ValueError) as exc:
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"]) from exc
    if (
        vertical[0] >= vertical[1]
        or not math.isfinite(half_thickness)
        or half_thickness <= 0.0
    ):
        raise ArticulatedWorkspaceClearanceError(["sweep_parameter_invalid"])
    radius = math.hypot(endpoint[0] - hinge[0], endpoint[1] - hinge[1])
    if radius <= 0.0:
        raise ArticulatedWorkspaceClearanceError(["sweep_member_radius_invalid"])
    source_angle = math.atan2(endpoint[1] - hinge[1], endpoint[0] - hinge[0])

    stage = Usd.Stage.Open(str(collision), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ArticulatedWorkspaceClearanceError(
            ["sage_mesh_sweep_collision_usd_open_failed"]
        )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_triangles: list[
        tuple[str, list[tuple[list[list[float]], list[float], list[float]]]]
    ] = []
    for row in inventory["obstacles"]:
        prim_path = str(row["source_prim_path"])
        prim = stage.GetPrimAtPath(prim_path)
        if (
            not prim.IsValid()
            or not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            raise ArticulatedWorkspaceClearanceError(
                [f"sage_mesh_sweep_obstacle_prim_invalid:{prim_path}"]
            )
        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default()) or []
        counts = [int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        transform = xform_cache.GetLocalToWorldTransform(prim)
        world_points = [
            [float(value) for value in transform.Transform(point)]
            for point in local_points
        ]
        triangles: list[list[list[float]]] = []
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3:
                continue
            for offset in range(1, count - 1):
                triangles.append(
                    [
                        world_points[face[0]],
                        world_points[face[offset]],
                        world_points[face[offset + 1]],
                    ]
                )
        mesh_triangles.append(
            (
                prim_path,
                [
                    (
                        triangle,
                        [min(point[axis] for point in triangle) for axis in range(3)],
                        [max(point[axis] for point in triangle) for axis in range(3)],
                    )
                    for triangle in triangles
                ],
            )
        )

    z_center = 0.5 * (vertical[0] + vertical[1])
    half_extents = [0.5 * radius, half_thickness, 0.5 * (vertical[1] - vertical[0])]
    door_state_rows: list[dict[str, Any]] = []
    first_contact: dict[str, Any] | None = None
    for angle_degrees in states:
        angle = source_angle + math.radians(angle_degrees)
        radial = [math.cos(angle), math.sin(angle), 0.0]
        tangent = [-math.sin(angle), math.cos(angle), 0.0]
        center = [
            hinge[0] + 0.5 * radius * radial[0],
            hinge[1] + 0.5 * radius * radial[1],
            z_center,
        ]
        door_world_half_extents = [
            abs(radial[0]) * half_extents[0] + abs(tangent[0]) * half_extents[1],
            abs(radial[1]) * half_extents[0] + abs(tangent[1]) * half_extents[1],
            half_extents[2],
        ]
        door_minimum = [center[axis] - door_world_half_extents[axis] for axis in range(3)]
        door_maximum = [center[axis] + door_world_half_extents[axis] for axis in range(3)]

        def _prism_hits(
            triangle_rows: list[tuple[list[list[float]], list[float], list[float]]],
        ) -> bool:
            for triangle, triangle_minimum, triangle_maximum in triangle_rows:
                if any(
                    min(door_maximum[axis], triangle_maximum[axis])
                    < max(door_minimum[axis], triangle_minimum[axis])
                    for axis in range(3)
                ):
                    continue
                local_triangle = []
                for point in triangle:
                    relative = [point[index] - center[index] for index in range(3)]
                    local_triangle.append(
                        [_dot(relative, radial), _dot(relative, tangent), relative[2]]
                    )
                if _triangle_intersects_axis_aligned_box(local_triangle, half_extents):
                    return True
            return False

        sage_contacts = sorted(
            prim_path
            for prim_path, triangle_rows in mesh_triangles
            if _prism_hits(triangle_rows)
        )
        box_contacts = [
            {"label": box["label"], "obstacle_class": box["obstacle_class"]}
            for box in boxes
            if _prism_hits(box["triangles"])
        ]
        clear = not sage_contacts and not box_contacts
        if not clear and first_contact is None:
            if sage_contacts:
                first_contact = {
                    "angle_degrees": angle_degrees,
                    "source": sage_contacts[0],
                    "obstacle_class": "sage_static_scene",
                }
            else:
                first_contact = {
                    "angle_degrees": angle_degrees,
                    "source": box_contacts[0]["label"],
                    "obstacle_class": box_contacts[0]["obstacle_class"],
                }
        door_state_rows.append(
            {
                "angle_degrees": angle_degrees,
                "sage_contact_prim_paths": sage_contacts,
                "static_box_contacts": box_contacts,
                "clear": clear,
            }
        )

    bound_classes = sorted({box["obstacle_class"] for box in boxes})
    result: dict[str, Any] = {
        "schema_version": DOOR_STATE_CLEARANCE_SCHEMA_VERSION,
        "status": (
            "blocked_by_door_state_contact"
            if first_contact
            else "door_state_matrix_clearance_candidate_only"
        ),
        "source": {
            "sage_collision_usd_sha256": _sha256(collision),
            "obstacle_inventory_receipt_digest": inventory["receipt_digest"],
        },
        "sweep": {
            "hinge_origin_world_m": hinge,
            "closed_endpoint_world_m": endpoint,
            "member_vertical_interval_m": vertical,
            "member_radius_m": radius,
            "member_half_thickness_m": half_thickness,
            "required_maximum_angle_degrees": required_maximum,
        },
        "door_state_rows": door_state_rows,
        "static_box_obstacles": [
            {
                "label": box["label"],
                "obstacle_class": box["obstacle_class"],
                "aabb_min": box["aabb_min"],
                "aabb_max": box["aabb_max"],
            }
            for box in boxes
        ],
        "static_obstacle_classes_bound": bound_classes,
        "first_contact": first_contact,
        "claim_boundary": {
            "triangle_prism_intersection_tested": True,
            "full_stage_inventory_is_bound_broadphase": True,
            "clear_result_is_not_native_dynamic_qualification": True,
            "replacement_self_geometry_bound": any(
                obstacle_class in {"replacement_body", "replacement_lower_door"}
                for obstacle_class in bound_classes
            ),
            "franka_base_bound": "franka_base" in bound_classes,
            "ik_or_contact_qualified": False,
        },
        "receipt_digest": "",
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    return result


def validate_articulated_workspace_clearance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a retained clearance receipt without rerunning geometry."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("sweep_clearance_schema_invalid")
    if payload.get("status") not in {
        "blocked_by_observed_obstacle",
        "clearance_candidate_only",
    }:
        errors.append("sweep_clearance_status_invalid")
    collision = payload.get("first_collision")
    if payload.get("status") == "blocked_by_observed_obstacle" and not isinstance(
        collision, Mapping
    ):
        errors.append("sweep_clearance_collision_missing")
    if payload.get("status") == "clearance_candidate_only" and collision is not None:
        errors.append("sweep_clearance_unexpected_collision")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("sweep_clearance_digest_invalid")
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)
    return payload


def validate_door_state_clearance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a retained twelve-door-state clearance receipt."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != DOOR_STATE_CLEARANCE_SCHEMA_VERSION:
        errors.append("door_state_clearance_schema_invalid")
    status = payload.get("status")
    if status not in {
        "blocked_by_door_state_contact",
        "door_state_matrix_clearance_candidate_only",
    }:
        errors.append("door_state_clearance_status_invalid")
    rows = payload.get("door_state_rows")
    if not isinstance(rows, list) or len(rows) < DOOR_STATE_MINIMUM_COUNT:
        errors.append("door_state_clearance_rows_invalid")
        rows = []
    first_contact = payload.get("first_contact")
    any_unclear = any(
        not row.get("clear") for row in rows if isinstance(row, Mapping)
    )
    if status == "blocked_by_door_state_contact" and (
        not isinstance(first_contact, Mapping) or not any_unclear
    ):
        errors.append("door_state_clearance_contact_missing")
    if status == "door_state_matrix_clearance_candidate_only" and (
        first_contact is not None or any_unclear
    ):
        errors.append("door_state_clearance_unexpected_contact")
    boundary = payload.get("claim_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("triangle_prism_intersection_tested") is not True
        or boundary.get("clear_result_is_not_native_dynamic_qualification")
        is not True
    ):
        errors.append("door_state_clearance_exact_evidence_missing")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("door_state_clearance_digest_invalid")
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)
    return payload


def validate_sage_mesh_sweep(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact SAGE triangle sweep before harness admission."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SAGE_MESH_SWEEP_SCHEMA_VERSION:
        errors.append("sage_mesh_sweep_schema_invalid")
    status = payload.get("status")
    if status not in {
        "blocked_by_exact_sage_mesh_contact",
        "exact_sage_mesh_clearance_candidate_only",
    }:
        errors.append("sage_mesh_sweep_status_invalid")
    collision = payload.get("first_collision")
    paths = payload.get("collision_prim_paths")
    if status == "blocked_by_exact_sage_mesh_contact" and (
        not isinstance(collision, Mapping) or not isinstance(paths, list) or not paths
    ):
        errors.append("sage_mesh_sweep_collision_missing")
    if status == "exact_sage_mesh_clearance_candidate_only" and (
        collision is not None or paths != []
    ):
        errors.append("sage_mesh_sweep_unexpected_collision")
    boundary = payload.get("claim_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("triangle_prism_intersection_tested") is not True
        or boundary.get("full_stage_inventory_is_bound_broadphase") is not True
    ):
        errors.append("sage_mesh_sweep_exact_evidence_missing")
    if payload.get("receipt_digest") != canonical_digest(
        payload, digest_field="receipt_digest"
    ):
        errors.append("sage_mesh_sweep_digest_invalid")
    if errors:
        raise ArticulatedWorkspaceClearanceError(errors)
    return payload


__all__ = [
    "ArticulatedWorkspaceClearanceError",
    "SCHEMA_VERSION",
    "SAGE_OBSTACLE_INVENTORY_SCHEMA_VERSION",
    "SAGE_MESH_SWEEP_SCHEMA_VERSION",
    "evaluate_revolute_member_sweep",
    "evaluate_frozen_door_state_clearance",
    "evaluate_revolute_member_sweep_against_sage_meshes",
    "inventory_sage_sweep_obstacles",
    "load_bound_collision_obstacle",
    "obstacles_from_sage_sweep_inventory",
    "validate_articulated_workspace_clearance",
    "validate_door_state_clearance",
    "validate_sage_mesh_sweep",
]
