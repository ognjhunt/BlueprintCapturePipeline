"""General, task-neutral articulation graph validation for ADP task instances.

The contract describes the complete replacement-asset mechanism independently
from any scene, semantic object class, or simulator implementation.  It does
not infer a mechanism from labels or generated geometry: callers must bind the
graph to a separately qualified asset and retain observed-versus-generated
provenance in the task freeze.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "adp_articulation_graph.v1"
JOINT_ROLES = frozenset({"target", "dependent", "passive", "locked"})
JOINT_TYPES = frozenset({"revolute", "prismatic", "continuous", "fixed"})
DRIVE_TYPES = frozenset({"force", "acceleration", "none"})


class ArticulationGraphContractError(ValueError):
    """Stable, sorted articulation-graph validation failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector3(value: Any) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        return None
    result = [_finite(item) for item in value]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def _interval(value: Any, *, allow_equal: bool = False) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        return None
    lower = _finite(value[0])
    upper = _finite(value[1])
    if lower is None or upper is None:
        return None
    if lower > upper or (lower == upper and not allow_equal):
        return None
    return [lower, upper]


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _has_cycle(parent_by_child: Mapping[str, str]) -> bool:
    for first in parent_by_child:
        seen: set[str] = set()
        current = first
        while current in parent_by_child:
            if current in seen:
                return True
            seen.add(current)
            current = parent_by_child[current]
    return False


def validate_articulation_graph(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a complete link/joint graph without a size cap."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("articulation_graph_schema_invalid")

    links = _rows(payload.get("links"))
    if not links or len(links) != len(payload.get("links") or []):
        errors.append("articulation_graph_links_invalid")
    link_ids = [str(row.get("link_id") or "") for row in links]
    if any(not link_id for link_id in link_ids) or len(link_ids) != len(set(link_ids)):
        errors.append("articulation_graph_link_ids_invalid")
    roots = [row for row in links if row.get("is_root") is True]
    if len(roots) != 1:
        errors.append("articulation_graph_root_count_invalid")
    normalized_links = [
        {
            "link_id": str(row.get("link_id") or ""),
            "is_root": row.get("is_root") is True,
            "semantic_role": str(row.get("semantic_role") or ""),
        }
        for row in links
    ]
    if any(not row["semantic_role"] for row in normalized_links):
        errors.append("articulation_graph_link_semantic_role_missing")

    joints = _rows(payload.get("joints"))
    if len(joints) != len(payload.get("joints") or []):
        errors.append("articulation_graph_joints_invalid")
    joint_ids = [str(row.get("joint_id") or "") for row in joints]
    if any(not joint_id for joint_id in joint_ids) or len(joint_ids) != len(set(joint_ids)):
        errors.append("articulation_graph_joint_ids_invalid")
    normalized_joints: list[dict[str, Any]] = []
    parent_by_child: dict[str, str] = {}
    for index, row in enumerate(joints):
        joint_id = str(row.get("joint_id") or f"joint_{index}")
        parent = str(row.get("parent_link_id") or "")
        child = str(row.get("child_link_id") or "")
        joint_type = str(row.get("joint_type") or "")
        role = str(row.get("role") or "")
        if parent not in link_ids or child not in link_ids or parent == child:
            errors.append(f"articulation_graph_joint_links_invalid:{joint_id}")
        if child in parent_by_child:
            errors.append(f"articulation_graph_child_has_multiple_parents:{child}")
        elif parent and child:
            parent_by_child[child] = parent
        if joint_type not in JOINT_TYPES:
            errors.append(f"articulation_graph_joint_type_invalid:{joint_id}")
        if role not in JOINT_ROLES:
            errors.append(f"articulation_graph_joint_role_invalid:{joint_id}")
        if joint_type == "fixed" and role != "locked":
            errors.append(f"articulation_graph_fixed_joint_not_locked:{joint_id}")

        axis = _vector3(row.get("axis"))
        if axis is None or (
            joint_type != "fixed"
            and math.sqrt(sum(component * component for component in axis)) <= 1e-12
        ):
            errors.append(f"articulation_graph_joint_axis_invalid:{joint_id}")
            axis = [0.0, 0.0, 0.0]
        limits = _interval(row.get("limits"), allow_equal=joint_type == "fixed")
        if limits is None:
            errors.append(f"articulation_graph_joint_limits_invalid:{joint_id}")
            limits = [0.0, 0.0]
        reset = _finite(row.get("reset_position"))
        if reset is None or not (limits[0] <= reset <= limits[1]):
            errors.append(f"articulation_graph_joint_reset_invalid:{joint_id}")
            reset = 0.0
        reset_tolerance = _finite(row.get("reset_tolerance"))
        if reset_tolerance is None or reset_tolerance <= 0.0:
            errors.append(f"articulation_graph_joint_reset_tolerance_invalid:{joint_id}")
            reset_tolerance = 0.0
        drive = row.get("drive")
        normalized_drive: dict[str, Any]
        if not isinstance(drive, Mapping):
            errors.append(f"articulation_graph_joint_drive_invalid:{joint_id}")
            normalized_drive = {
                "drive_type": "none",
                "stiffness": 0.0,
                "damping": 0.0,
                "maximum_force": 0.0,
            }
        else:
            drive_type = str(drive.get("drive_type") or "")
            stiffness = _finite(drive.get("stiffness"))
            damping = _finite(drive.get("damping"))
            maximum_force = _finite(drive.get("maximum_force"))
            if (
                drive_type not in DRIVE_TYPES
                or stiffness is None
                or stiffness < 0.0
                or damping is None
                or damping < 0.0
                or maximum_force is None
                or maximum_force < 0.0
            ):
                errors.append(f"articulation_graph_joint_drive_invalid:{joint_id}")
            normalized_drive = {
                "drive_type": drive_type,
                "stiffness": float(stiffness or 0.0),
                "damping": float(damping or 0.0),
                "maximum_force": float(maximum_force or 0.0),
            }
        dependency = row.get("dependency")
        normalized_dependency = None
        if role == "dependent":
            if not isinstance(dependency, Mapping):
                errors.append(f"articulation_graph_dependency_missing:{joint_id}")
            else:
                driver = str(dependency.get("driver_joint_id") or "")
                multiplier = _finite(dependency.get("multiplier"))
                offset = _finite(dependency.get("offset"))
                tolerance = _finite(dependency.get("tolerance"))
                if (
                    driver not in joint_ids
                    or driver == joint_id
                    or multiplier is None
                    or offset is None
                    or tolerance is None
                    or tolerance <= 0.0
                ):
                    errors.append(f"articulation_graph_dependency_invalid:{joint_id}")
                normalized_dependency = {
                    "driver_joint_id": driver,
                    "multiplier": float(multiplier or 0.0),
                    "offset": float(offset or 0.0),
                    "tolerance": float(tolerance or 0.0),
                }
        elif dependency is not None:
            errors.append(f"articulation_graph_dependency_on_wrong_role:{joint_id}")

        normalized_joints.append(
            {
                "joint_id": joint_id,
                "parent_link_id": parent,
                "child_link_id": child,
                "joint_type": joint_type,
                "role": role,
                "axis": axis,
                "limits": limits,
                "reset_position": reset,
                "reset_tolerance": reset_tolerance,
                "drive": normalized_drive,
                "dependency": normalized_dependency,
            }
        )

    if _has_cycle(parent_by_child):
        errors.append("articulation_graph_cycle_detected")
    if links and len(parent_by_child) != len(links) - 1:
        errors.append("articulation_graph_not_connected_tree")

    collision_pairs = _rows(payload.get("collision_pairs"))
    if len(collision_pairs) != len(payload.get("collision_pairs") or []):
        errors.append("articulation_graph_collision_pairs_invalid")
    normalized_pairs: list[dict[str, Any]] = []
    pair_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(collision_pairs):
        link_a = str(row.get("link_a") or "")
        link_b = str(row.get("link_b") or "")
        key = tuple(sorted((link_a, link_b)))
        if (
            link_a not in link_ids
            or link_b not in link_ids
            or link_a == link_b
            or key in pair_keys
            or not isinstance(row.get("collision_enabled"), bool)
        ):
            errors.append(f"articulation_graph_collision_pair_invalid:{index}")
        pair_keys.add(key)
        normalized_pairs.append(
            {
                "link_a": link_a,
                "link_b": link_b,
                "collision_enabled": row.get("collision_enabled") is True,
            }
        )

    target_joint_ids = [
        row["joint_id"] for row in normalized_joints if row["role"] == "target"
    ]
    if not target_joint_ids:
        errors.append("articulation_graph_target_joint_missing")
    predicate = payload.get("success_predicate")
    normalized_predicate: dict[str, Any]
    if not isinstance(predicate, Mapping):
        errors.append("articulation_graph_success_predicate_missing")
        normalized_predicate = {"combination": "all", "joint_intervals": {}}
    else:
        combination = str(predicate.get("combination") or "")
        intervals = predicate.get("joint_intervals")
        if combination != "all":
            errors.append("articulation_graph_success_combination_invalid")
        if not isinstance(intervals, Mapping) or set(intervals) != set(target_joint_ids):
            errors.append("articulation_graph_success_joint_set_invalid")
            intervals = {}
        normalized_intervals: dict[str, list[float]] = {}
        for joint_id, raw in intervals.items():
            interval = _interval(raw)
            joint = next(
                (item for item in normalized_joints if item["joint_id"] == joint_id),
                None,
            )
            if (
                interval is None
                or joint is None
                or interval[0] < joint["limits"][0]
                or interval[1] > joint["limits"][1]
                or interval[0] <= joint["reset_position"] <= interval[1]
            ):
                errors.append(f"articulation_graph_success_interval_invalid:{joint_id}")
            else:
                normalized_intervals[str(joint_id)] = interval
        normalized_predicate = {
            "combination": combination,
            "joint_intervals": normalized_intervals,
        }

    if errors:
        raise ArticulationGraphContractError(errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "links": normalized_links,
        "joints": normalized_joints,
        "collision_pairs": normalized_pairs,
        "success_predicate": normalized_predicate,
    }


__all__ = [
    "ArticulationGraphContractError",
    "DRIVE_TYPES",
    "JOINT_ROLES",
    "JOINT_TYPES",
    "SCHEMA_VERSION",
    "validate_articulation_graph",
]
