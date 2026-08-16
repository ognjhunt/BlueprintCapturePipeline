"""Deterministically admit Joint Agent topology before owned-core publication.

Joint Agent is a model-backed research preview.  Its candidate document is
therefore an input to this gate, never its own success receipt.  The gate binds
the candidate graph to a preregistered task-joint axis and an independently
computed moving-link bounds interval while allowing bounded non-task joints.

The gate reasons at LINK level, not raw-candidate level.  The released
optimizer hands the model one mesh split into anonymous per-component prims, so
a faithful model answer may describe one physical link with several candidates
(panel, handle, hinge hardware).  A deterministic geometry-only resolver groups
candidates into links using the source-asset receipt's connected-component
AABBs plus containment/adjacency heuristics; the preregistered joint-count
bounds then apply to distinct link-level joints.  The resolver never reads any
model- or manifest-authored link naming, so the model's answer is never fed
into its own grader.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "joint_agent_articulation_review.v2"

# Geometric slack used by the link-membership resolver and the target
# containment classification, expressed as a fraction of the containing
# interval's length.  Split prims and receipt components describe identical
# geometry, so only rounding-scale slack is required.
CONTAINMENT_TOLERANCE_FRACTION = 0.05
# A member whose vertical extent covers at least this fraction of the whole
# assembly's vertical extent is the assembly frame (the fixed body's own
# geometry claimed as a mover), never a sub-member carrier: it can neither
# absorb smaller members nor be absorbed.
FRAME_COEXTENSIVE_MINIMUM_FRACTION = 0.9
# A contained member is attached hardware of its carrier only when it is small
# relative to the carrier.  Two near-coincident bands are competing member
# claims that must stay distinct so the ambiguity gate can see them.
ABSORPTION_MAXIMUM_EXTENT_FRACTION = 0.5

MEASURED_BOUNDS_STATUS = "measured_from_optimized_usd"
UNMEASURED_BOUNDS_STATUS = "unmeasured"
FIXED_BODY_LINK_ID = "fixed_body"

# ``non_task_joint_mode`` values the gate understands.  The exempt mode admits
# bounded non-task candidates that still carry unresolved reason codes because
# the downstream run locks every non-task joint at its frozen reset and reads
# the lock back natively; the strict mode requires every candidate to be fully
# rigger-resolved.  Anything else fails closed.
NON_TASK_JOINT_MODE_EXEMPT = "locked_at_frozen_reset_with_native_readback"
NON_TASK_JOINT_MODE_STRICT = "require_fully_resolved"
NON_TASK_JOINT_MODES = frozenset(
    {NON_TASK_JOINT_MODE_EXEMPT, NON_TASK_JOINT_MODE_STRICT}
)


class JointAgentArticulationReviewError(ValueError):
    """Stable, sorted topology-review failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite_vector(value: Any, *, length: int) -> tuple[float, ...] | None:
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


def _normalized_axis(value: Any) -> tuple[float, float, float] | None:
    vector = _finite_vector(value, length=3)
    if vector is None:
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        return None
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise JointAgentArticulationReviewError([error]) from exc
    if not isinstance(cloned, dict):
        raise JointAgentArticulationReviewError([error])
    return cloned


def _clone_rows(value: Any, *, error: str) -> list[Any]:
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise JointAgentArticulationReviewError([error]) from exc
    if not isinstance(cloned, list):
        raise JointAgentArticulationReviewError([error])
    return cloned


Aabb = tuple[tuple[float, float, float], tuple[float, float, float]]


def _aabb(minimum: Any, maximum: Any) -> Aabb | None:
    low = _finite_vector(minimum, length=3)
    high = _finite_vector(maximum, length=3)
    if (
        low is None
        or high is None
        or any(low_v > high_v for low_v, high_v in zip(low, high, strict=True))
    ):
        return None
    return (low, high)  # type: ignore[return-value]


def _extent(aabb: Aabb, axis: int) -> float:
    return aabb[1][axis] - aabb[0][axis]


def _interval_within(
    inner: tuple[float, float], outer: tuple[float, float], tolerance: float
) -> bool:
    return inner[0] >= outer[0] - tolerance and inner[1] <= outer[1] + tolerance


def _axes_touch(member: Aabb, item: Aabb, axis: int, tolerance: float) -> bool:
    overlap = min(member[1][axis], item[1][axis]) - max(member[0][axis], item[0][axis])
    return overlap > -tolerance


def _member_claims(member: Aabb, item: Aabb) -> bool:
    """True when ``member``'s vertical band claims ``item``.

    The item's vertical interval must lie inside the member's vertical interval
    and the two footprints must touch in both horizontal axes, each within a
    tolerance proportional to the member's own extent.
    """

    vertical_tolerance = CONTAINMENT_TOLERANCE_FRACTION * max(_extent(member, 2), 0.0)
    if not _interval_within(
        (item[0][2], item[1][2]), (member[0][2], member[1][2]), vertical_tolerance
    ):
        return False
    return all(
        _axes_touch(
            member,
            item,
            axis,
            CONTAINMENT_TOLERANCE_FRACTION * max(_extent(member, axis), 0.0),
        )
        for axis in (0, 1)
    )


def _union_aabb(aabbs: Sequence[Aabb]) -> Aabb:
    return (
        tuple(min(aabb[0][axis] for aabb in aabbs) for axis in range(3)),
        tuple(max(aabb[1][axis] for aabb in aabbs) for axis in range(3)),
    )  # type: ignore[return-value]


def resolve_link_membership(
    *,
    member_aabbs: Mapping[str, Aabb],
    source_component_aabbs: Sequence[tuple[int, Aabb]],
) -> dict[str, Any]:
    """Group measured members into links using geometry only.

    ``member_aabbs`` maps candidate IDs to independently measured AABBs;
    ``source_component_aabbs`` are the source-asset receipt's connected
    components (never a manifest's link naming).  Members with mutually
    coincident bands are one link; a smaller member whose band lies inside a
    larger non-frame member's band is attached hardware of that member; the
    components no moving link claims are the fixed body.
    """

    identifiers = sorted(member_aabbs)
    parent: dict[str, str] = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(identifiers):
        for right in identifiers[index + 1 :]:
            left_aabb, right_aabb = member_aabbs[left], member_aabbs[right]
            if _member_claims(left_aabb, right_aabb) and _member_claims(
                right_aabb, left_aabb
            ):
                union(left, right)

    clusters: dict[str, dict[str, Any]] = {}
    for identifier in identifiers:
        root = find(identifier)
        cluster = clusters.setdefault(root, {"member_ids": [], "aabbs": []})
        cluster["member_ids"].append(identifier)
        cluster["aabbs"].append(member_aabbs[identifier])
    groups = [
        {
            "member_ids": sorted(cluster["member_ids"]),
            "aabb": _union_aabb(cluster["aabbs"]),
        }
        for cluster in clusters.values()
    ]

    component_aabbs = [aabb for _index, aabb in source_component_aabbs]
    assembly_aabb = (
        _union_aabb(component_aabbs)
        if component_aabbs
        else (_union_aabb([group["aabb"] for group in groups]) if groups else None)
    )
    assembly_vertical_extent = _extent(assembly_aabb, 2) if assembly_aabb else 0.0

    def frame_coextensive(aabb: Aabb) -> bool:
        return (
            assembly_vertical_extent > 0.0
            and _extent(aabb, 2)
            >= FRAME_COEXTENSIVE_MINIMUM_FRACTION * assembly_vertical_extent
        )

    changed = True
    while changed:
        changed = False
        groups.sort(key=lambda group: (_extent(group["aabb"], 2), group["member_ids"]))
        for index, group in enumerate(groups):
            if frame_coextensive(group["aabb"]):
                continue
            containers = [
                other
                for other in groups
                if other is not group
                and not frame_coextensive(other["aabb"])
                and _extent(group["aabb"], 2)
                <= ABSORPTION_MAXIMUM_EXTENT_FRACTION * _extent(other["aabb"], 2)
                and _member_claims(other["aabb"], group["aabb"])
            ]
            if containers:
                target = min(
                    containers,
                    key=lambda other: (_extent(other["aabb"], 2), other["member_ids"]),
                )
                target["member_ids"] = sorted(target["member_ids"] + group["member_ids"])
                target["aabb"] = _union_aabb([target["aabb"], group["aabb"]])
                del groups[index]
                changed = True
                break

    groups.sort(key=lambda group: (group["aabb"][0], group["member_ids"]))
    links: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        links.append(
            {
                "link_id": f"link_{index:02d}",
                "member_candidate_ids": group["member_ids"],
                "aabb_min": list(group["aabb"][0]),
                "aabb_max": list(group["aabb"][1]),
                "frame_coextensive": frame_coextensive(group["aabb"]),
                "component_indices": [],
                "_aabb": group["aabb"],
            }
        )

    member_link: dict[str, str] = {}
    for link in links:
        for identifier in link["member_candidate_ids"]:
            member_link[identifier] = link["link_id"]

    fixed_component_indices: list[int] = []
    for component_index, component_aabb in sorted(source_component_aabbs):
        claimants = [
            link
            for link in links
            if not link["frame_coextensive"]
            and _member_claims(link["_aabb"], component_aabb)
        ]
        if claimants:
            owner = min(
                claimants,
                key=lambda link: (_extent(link["_aabb"], 2), link["link_id"]),
            )
            owner["component_indices"].append(component_index)
        else:
            fixed_component_indices.append(component_index)

    return {
        "membership_basis": "geometry_only",
        "links": links,
        "member_link": member_link,
        "fixed_component_indices": fixed_component_indices,
        "articulation_root_count": 1 if fixed_component_indices else 0,
        "assembly_aabb": assembly_aabb,
        "assembly_vertical_extent": assembly_vertical_extent,
    }


def _resolve_member_link(
    aabb: Aabb, *, links: Sequence[Mapping[str, Any]], assembly_aabb: Aabb | None
) -> str | None:
    claimants = [
        link
        for link in links
        if not link["frame_coextensive"] and _member_claims(link["_aabb"], aabb)
    ]
    if claimants:
        return min(
            claimants, key=lambda link: (_extent(link["_aabb"], 2), link["link_id"])
        )["link_id"]
    if assembly_aabb is not None:
        tolerance = CONTAINMENT_TOLERANCE_FRACTION * max(
            _extent(assembly_aabb, 2), 0.0
        )
        if _interval_within(
            (aabb[0][2], aabb[1][2]),
            (assembly_aabb[0][2], assembly_aabb[1][2]),
            tolerance,
        ):
            return FIXED_BODY_LINK_ID
    return None


def _link_cycle_members(edges: Mapping[str, set[str]]) -> set[str]:
    """Return every link that participates in a parent-graph cycle."""

    in_cycle: set[str] = set()
    for start in sorted(edges):
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            for successor in sorted(edges.get(node, set())):
                if successor == start:
                    in_cycle.update(path)
                elif successor not in path:
                    stack.append((successor, path + (successor,)))
    return in_cycle


def review_joint_agent_articulation(
    *,
    candidates_document: Mapping[str, Any],
    candidate_bounds: Mapping[str, Any],
    review_contract: Mapping[str, Any],
    source_components: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Admit the commanded task joints inside a bounded multi-joint assembly.

    ``candidate_bounds`` must be computed from the optimized USD by the runtime;
    each candidate ID maps to ``{"aabb_min": [x,y,z], "aabb_max": [x,y,z]}`` or
    a typed ``{"status": "unmeasured", ...}`` record, optionally carrying a
    measured ``fixed_parent_bounds`` for the candidate's declared parent prim.
    ``source_components`` are the source-asset receipt's connected components
    (``aabb_min_asset_m``/``aabb_max_asset_m``); they anchor link membership and
    the fixed-body complement.  A caller-provided label or model confidence
    alone can never select the task joint.
    """

    document = _clone(candidates_document, error="joint_candidates_not_json")
    bounds = _clone(candidate_bounds, error="joint_candidate_bounds_not_json")
    contract = _clone(review_contract, error="joint_review_contract_not_json")
    errors: list[str] = []

    if source_components is None:
        errors.append("joint_review_source_components_missing")
        components_document: list[Any] = []
    else:
        components_document = _clone_rows(
            source_components, error="joint_review_source_components_not_json"
        )
    component_aabbs: list[tuple[int, Aabb]] = []
    seen_component_indices: set[int] = set()
    for position, component in enumerate(components_document):
        aabb = (
            _aabb(
                component.get("aabb_min_asset_m"), component.get("aabb_max_asset_m")
            )
            if isinstance(component, Mapping)
            else None
        )
        index = component.get("component_index") if isinstance(component, Mapping) else None
        if (
            aabb is None
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index in seen_component_indices
        ):
            errors.append(f"joint_review_source_component_invalid:{position}")
            continue
        seen_component_indices.add(index)
        component_aabbs.append((index, aabb))
    if source_components is not None and not component_aabbs:
        errors.append("joint_review_source_components_missing")

    if document.get("schema_version") != "joint-agent-stage2-v0":
        errors.append("joint_candidates_schema_invalid")
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
        errors.append("joint_candidates_list_invalid")
    summary = document.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("joint_candidates_summary_missing")
    elif summary.get("candidate_count") != len(candidates):
        errors.append("joint_candidates_summary_count_mismatch")

    max_joints = contract.get("maximum_assembly_joint_count")
    if isinstance(max_joints, bool) or not isinstance(max_joints, int) or max_joints < 1:
        errors.append("joint_review_maximum_joint_count_invalid")
        max_joints = 0
    min_joints = contract.get("minimum_assembly_joint_count")
    if (
        isinstance(min_joints, bool)
        or not isinstance(min_joints, int)
        or min_joints < 1
        or (max_joints >= 1 and min_joints > max_joints)
    ):
        errors.append("joint_review_minimum_joint_count_invalid")
        min_joints = 1
    commanded_task_joints = contract.get("commanded_task_joint_count")
    if (
        isinstance(commanded_task_joints, bool)
        or not isinstance(commanded_task_joints, int)
        or commanded_task_joints < 1
    ):
        errors.append("joint_review_commanded_task_joint_count_invalid")
        commanded_task_joints = 1
    required_roots = contract.get("required_articulation_root_count")
    if (
        isinstance(required_roots, bool)
        or not isinstance(required_roots, int)
        or required_roots < 1
    ):
        errors.append("joint_review_required_root_count_invalid")
        required_roots = 1
    non_task_mode = contract.get("non_task_joint_mode")
    if non_task_mode not in NON_TASK_JOINT_MODES:
        errors.append("joint_review_non_task_joint_mode_invalid")
        non_task_mode = NON_TASK_JOINT_MODE_STRICT
    non_task_exempt = non_task_mode == NON_TASK_JOINT_MODE_EXEMPT
    non_task_tolerance = contract.get("non_task_joint_motion_tolerance")
    if (
        isinstance(non_task_tolerance, bool)
        or not isinstance(non_task_tolerance, (int, float))
        or not math.isfinite(float(non_task_tolerance))
        or float(non_task_tolerance) < 0.0
    ):
        errors.append("joint_review_non_task_joint_motion_tolerance_invalid")
        non_task_tolerance = None
    else:
        non_task_tolerance = float(non_task_tolerance)
    extent_band = _finite_vector(
        contract.get("target_member_extent_ratio_band"), length=2
    )
    if extent_band is None or not 0.0 < extent_band[0] <= extent_band[1]:
        errors.append("joint_review_extent_ratio_band_invalid")
        extent_band = (1.0, 0.0)

    allowed = contract.get("allowed_joint_types")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(item not in {"revolute", "prismatic"} for item in allowed)
    ):
        errors.append("joint_review_allowed_types_invalid")
        allowed_types: set[str] = set()
    else:
        allowed_types = set(allowed)
    target_type = contract.get("target_joint_type")
    if target_type not in allowed_types:
        errors.append("joint_review_target_type_invalid")
    target_axis = _normalized_axis(contract.get("target_axis_world"))
    if target_axis is None:
        errors.append("joint_review_target_axis_invalid")
    axis_abs_dot_min = contract.get("target_axis_absolute_dot_minimum")
    if (
        isinstance(axis_abs_dot_min, bool)
        or not isinstance(axis_abs_dot_min, (int, float))
        or not 0.0 < float(axis_abs_dot_min) <= 1.0
    ):
        errors.append("joint_review_axis_threshold_invalid")
        axis_abs_dot_min = 2.0
    projection_constraints = contract.get("target_member_projection_constraints")
    if not isinstance(projection_constraints, list) or not projection_constraints:
        errors.append("joint_review_projection_constraints_invalid")
        projection_constraints = []
    normalized_constraints: list[
        tuple[tuple[float, float, float], tuple[float, float], float]
    ] = []
    for constraint_index, constraint in enumerate(projection_constraints):
        axis = (
            _normalized_axis(constraint.get("axis_world"))
            if isinstance(constraint, Mapping)
            else None
        )
        interval = (
            _finite_vector(constraint.get("interval_m"), length=2)
            if isinstance(constraint, Mapping)
            else None
        )
        minimum_overlap = (
            constraint.get("minimum_overlap_fraction")
            if isinstance(constraint, Mapping)
            else None
        )
        if (
            axis is None
            or interval is None
            or interval[0] >= interval[1]
            or isinstance(minimum_overlap, bool)
            or not isinstance(minimum_overlap, (int, float))
            or not 0.0 < float(minimum_overlap) <= 1.0
        ):
            errors.append(
                f"joint_review_projection_constraint_invalid:{constraint_index}"
            )
            continue
        normalized_constraints.append((axis, interval, float(minimum_overlap)))

    retained: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            errors.append(f"joint_candidate_invalid:{index}")
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_ids:
            errors.append("joint_candidate_id_missing_or_duplicate")
            continue
        seen_ids.add(candidate_id)
        joint_type = str(candidate.get("joint_type_hint") or "")
        type_admitted = joint_type in allowed_types
        unresolved_codes = candidate.get("unresolved_reason_codes")
        if unresolved_codes in ([], None):
            unresolved_codes = []
        elif isinstance(unresolved_codes, list):
            unresolved_codes = sorted(str(code) for code in unresolved_codes)
        else:
            unresolved_codes = ["unresolved_reason_codes_not_a_list"]
        rigger_ready = (
            candidate.get("review_status") == "ready_for_rigger_input"
            and not unresolved_codes
        )
        moving_prims = [
            str(item) for item in candidate.get("moving_part_prims") or []
        ]

        record = bounds.get(candidate_id)
        record = record if isinstance(record, Mapping) else None
        aabb = (
            _aabb(record.get("aabb_min"), record.get("aabb_max"))
            if record is not None
            else None
        )
        unmeasured = (
            record is not None
            and aabb is None
            and record.get("status") == UNMEASURED_BOUNDS_STATUS
            and record.get("aabb_min") is None
            and record.get("aabb_max") is None
        )
        if not moving_prims and record is not None:
            moving_prims = [
                str(item) for item in record.get("moving_part_prims") or []
            ]

        parent_prim = str(candidate.get("fixed_parent_prim") or "")
        parent_source = str(candidate.get("parent_resolution_source") or "")
        parent_record = (
            record.get("fixed_parent_bounds") if record is not None else None
        )
        parent_record = parent_record if isinstance(parent_record, Mapping) else None
        parent_aabb = (
            _aabb(parent_record.get("aabb_min"), parent_record.get("aabb_max"))
            if parent_record is not None
            else None
        )

        if aabb is None and not unmeasured:
            errors.append(f"joint_candidate_bounds_invalid:{candidate_id}")
            continue
        if unmeasured:
            if non_task_exempt:
                reason = str(record.get("unmeasured_reason") or UNMEASURED_BOUNDS_STATUS)
                dropped.append(
                    {
                        "candidate_id": candidate_id,
                        "drop_reason_code": f"unmeasured_candidate_bounds:{reason}",
                    }
                )
            else:
                errors.append(f"joint_candidate_unmeasured:{candidate_id}")
            continue
        if not type_admitted:
            if non_task_exempt:
                dropped.append(
                    {
                        "candidate_id": candidate_id,
                        "drop_reason_code": f"unadmitted_joint_type:{joint_type}",
                    }
                )
            else:
                errors.append(f"joint_candidate_type_not_admitted:{candidate_id}")
            continue
        if not non_task_exempt:
            if candidate.get("review_status") != "ready_for_rigger_input":
                errors.append(f"joint_candidate_not_rigger_ready:{candidate_id}")
            if unresolved_codes:
                errors.append(f"joint_candidate_unresolved:{candidate_id}")

        retained.append(
            {
                "candidate_id": candidate_id,
                "joint_type": joint_type,
                "review_status": str(candidate.get("review_status") or ""),
                "rigger_ready": rigger_ready,
                "unresolved_reason_codes": unresolved_codes,
                "moving_part_prims": moving_prims,
                "aabb": aabb,
                "axis": _normalized_axis(candidate.get("motion_axis_world")),
                "fixed_parent_prim": parent_prim,
                "parent_resolution_source": parent_source,
                "parent_aabb": parent_aabb,
            }
        )

    membership = resolve_link_membership(
        member_aabbs={entry["candidate_id"]: entry["aabb"] for entry in retained},
        source_component_aabbs=component_aabbs,
    )
    links = membership["links"]
    member_link: Mapping[str, str] = membership["member_link"]
    assembly_aabb = membership["assembly_aabb"]

    link_joints = sorted(
        {
            (member_link[entry["candidate_id"]], entry["joint_type"])
            for entry in retained
        }
    )
    if not min_joints <= len(link_joints) <= max_joints:
        errors.append("joint_candidate_count_outside_preregistered_bounds")
    if (
        source_components is not None
        and component_aabbs
        and membership["articulation_root_count"] != required_roots
    ):
        errors.append("joint_review_articulation_root_count_mismatch")

    articulation_root_prims = sorted(
        {
            "/" + prim.split("/")[1]
            for entry in retained
            for prim in entry["moving_part_prims"]
            if prim.startswith("/") and len(prim.split("/")) > 1 and prim.split("/")[1]
        }
    ) + ["/"]

    for entry in retained:
        violations: list[str] = []
        parent_prim = entry["fixed_parent_prim"]
        entry["parent_link_id"] = None
        if not parent_prim:
            violations.append("parent_missing")
        else:
            if not entry["parent_resolution_source"]:
                violations.append("parent_resolution_source_missing")
            if parent_prim in articulation_root_prims:
                violations.append("parent_is_articulation_root")
            if parent_prim in entry["moving_part_prims"]:
                violations.append("parent_self")
            if entry["parent_aabb"] is None:
                violations.append("parent_unmeasured")
            elif "parent_is_articulation_root" not in violations:
                parent_link = _resolve_member_link(
                    entry["parent_aabb"], links=links, assembly_aabb=assembly_aabb
                )
                entry["parent_link_id"] = parent_link
                if parent_link is None:
                    violations.append("parent_unresolvable")
                elif parent_link == member_link[entry["candidate_id"]]:
                    violations.append("parent_self")
        entry["parent_binding_violations"] = violations

    parent_edges: dict[str, set[str]] = {}
    for entry in retained:
        parent_link = entry["parent_link_id"]
        own_link = member_link[entry["candidate_id"]]
        # A self-parent is already a typed per-candidate violation; only
        # cross-link parent claims can form a kinematic cycle.
        if (
            parent_link is not None
            and parent_link != FIXED_BODY_LINK_ID
            and parent_link != own_link
        ):
            parent_edges.setdefault(own_link, set()).add(parent_link)
    cycle_links = _link_cycle_members(parent_edges)
    for entry in retained:
        if member_link[entry["candidate_id"]] in cycle_links:
            entry["parent_binding_violations"].append("parent_cycle")

    for entry in retained:
        overlap_fractions: list[float] = []
        extent_ratios: list[float] = []
        contained_flags: list[bool] = []
        center = tuple(
            (low + high) * 0.5
            for low, high in zip(entry["aabb"][0], entry["aabb"][1], strict=True)
        )
        half_extent = tuple(
            (high - low) * 0.5
            for low, high in zip(entry["aabb"][0], entry["aabb"][1], strict=True)
        )
        for selector_axis, target_interval, _threshold in normalized_constraints:
            center_projection = sum(
                value * axis_component
                for value, axis_component in zip(center, selector_axis, strict=True)
            )
            radius = sum(
                extent * abs(axis_component)
                for extent, axis_component in zip(
                    half_extent, selector_axis, strict=True
                )
            )
            candidate_interval = (
                center_projection - radius,
                center_projection + radius,
            )
            interval_length = target_interval[1] - target_interval[0]
            overlap = max(
                0.0,
                min(candidate_interval[1], target_interval[1])
                - max(candidate_interval[0], target_interval[0]),
            )
            overlap_fractions.append(overlap / interval_length)
            extent_ratios.append((2.0 * radius) / interval_length)
            contained_flags.append(
                _interval_within(
                    candidate_interval,
                    target_interval,
                    CONTAINMENT_TOLERANCE_FRACTION * interval_length,
                )
            )
        axis_dot = (
            abs(
                sum(
                    left * right
                    for left, right in zip(entry["axis"], target_axis, strict=True)
                )
            )
            if entry["axis"] is not None and target_axis is not None
            else None
        )
        projection_match = bool(normalized_constraints) and all(
            overlap >= threshold
            for overlap, (_axis, _interval, threshold) in zip(
                overlap_fractions, normalized_constraints, strict=True
            )
        )
        extent_within_band = bool(normalized_constraints) and all(
            extent_band[0] <= ratio <= extent_band[1] for ratio in extent_ratios
        )
        entry["axis_absolute_dot"] = axis_dot
        entry["target_projection_overlap_fractions"] = overlap_fractions or None
        entry["target_extent_ratios"] = extent_ratios or None
        entry["target_extent_ratio_within_band"] = extent_within_band
        entry["contained_by_target_intervals"] = (
            all(contained_flags) if contained_flags else False
        )
        entry["target_match"] = (
            entry["joint_type"] == target_type
            and axis_dot is not None
            and axis_dot >= float(axis_abs_dot_min)
            and projection_match
            and extent_within_band
        )

    def link_contained_by_target(link: Mapping[str, Any]) -> bool:
        aabb: Aabb = link["_aabb"]
        center = tuple(
            (low + high) * 0.5 for low, high in zip(aabb[0], aabb[1], strict=True)
        )
        half_extent = tuple(
            (high - low) * 0.5 for low, high in zip(aabb[0], aabb[1], strict=True)
        )
        if not normalized_constraints:
            return False
        for selector_axis, target_interval, _threshold in normalized_constraints:
            center_projection = sum(
                value * axis_component
                for value, axis_component in zip(center, selector_axis, strict=True)
            )
            radius = sum(
                extent * abs(axis_component)
                for extent, axis_component in zip(
                    half_extent, selector_axis, strict=True
                )
            )
            if not _interval_within(
                (center_projection - radius, center_projection + radius),
                target_interval,
                CONTAINMENT_TOLERANCE_FRACTION
                * (target_interval[1] - target_interval[0]),
            ):
                return False
        return True

    matched_links = sorted(
        {
            member_link[entry["candidate_id"]]
            for entry in retained
            if entry["target_match"]
        }
    )
    containment_disambiguation_applied = False
    rejected_containing_link_ids: list[str] = []
    selected_links = matched_links
    if len(matched_links) > commanded_task_joints:
        links_by_id = {link["link_id"]: link for link in links}
        contained = [
            link_id
            for link_id in matched_links
            if link_contained_by_target(links_by_id[link_id])
        ]
        if len(contained) == commanded_task_joints:
            containment_disambiguation_applied = True
            rejected_containing_link_ids = sorted(
                set(matched_links) - set(contained)
            )
            selected_links = contained
    if len(selected_links) != commanded_task_joints:
        if commanded_task_joints == 1:
            errors.append("exactly_one_task_joint_not_resolved")
        else:
            errors.append(
                "commanded_task_joint_count_not_resolved:"
                f"observed={len(selected_links)}:commanded={commanded_task_joints}"
            )
        selected_links = []

    task_candidate_ids: list[str] = []
    for entry in retained:
        is_task = (
            entry["target_match"]
            and member_link[entry["candidate_id"]] in selected_links
        )
        entry["task_candidate"] = is_task
        if not is_task:
            continue
        task_candidate_ids.append(entry["candidate_id"])
        if entry["review_status"] != "ready_for_rigger_input":
            errors.append(f"joint_candidate_not_rigger_ready:{entry['candidate_id']}")
        if entry["unresolved_reason_codes"]:
            errors.append(f"joint_candidate_unresolved:{entry['candidate_id']}")
        for violation in entry["parent_binding_violations"]:
            errors.append(f"joint_task_candidate_{violation}:{entry['candidate_id']}")
    task_candidate_ids.sort()

    if not non_task_exempt:
        for entry in retained:
            if entry["task_candidate"]:
                continue
            for violation in entry["parent_binding_violations"]:
                errors.append(f"joint_candidate_{violation}:{entry['candidate_id']}")

    if errors:
        raise JointAgentArticulationReviewError(errors)

    rows = [
        {
            "candidate_id": entry["candidate_id"],
            "joint_type": entry["joint_type"],
            "link_id": member_link[entry["candidate_id"]],
            "axis_absolute_dot": entry["axis_absolute_dot"],
            "target_projection_overlap_fractions": entry[
                "target_projection_overlap_fractions"
            ],
            "target_extent_ratios": entry["target_extent_ratios"],
            "target_extent_ratio_within_band": entry["target_extent_ratio_within_band"],
            "contained_by_target_intervals": entry["contained_by_target_intervals"],
            "target_match": entry["target_match"],
            "task_candidate": entry["task_candidate"],
            "rigger_ready": entry["rigger_ready"],
            "unresolved_reason_codes": entry["unresolved_reason_codes"],
            "non_task_exemption_applied": (
                non_task_exempt and not entry["task_candidate"]
            ),
            "fixed_parent_prim": entry["fixed_parent_prim"] or None,
            "parent_resolution_source": entry["parent_resolution_source"] or None,
            "parent_link_id": entry["parent_link_id"],
            "parent_binding_violations": sorted(
                set(entry["parent_binding_violations"])
            ),
        }
        for entry in sorted(retained, key=lambda entry: entry["candidate_id"])
    ]
    target_link_id = selected_links[0] if len(selected_links) == 1 else None
    retained_ids = {entry["candidate_id"] for entry in retained}

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_for_owned_core_topology_publication",
        "candidate_document_digest": canonical_digest(document),
        "candidate_bounds_digest": canonical_digest(bounds),
        "review_contract_digest": canonical_digest(contract),
        "source_components_digest": canonical_digest(
            {"connected_components": components_document}
        ),
        "raw_candidate_count": len(candidates),
        "assembly_joint_count": len(link_joints),
        "link_joints": [
            {"link_id": link_id, "joint_type": joint_type}
            for link_id, joint_type in link_joints
        ],
        "links": [
            {key: value for key, value in link.items() if key != "_aabb"}
            for link in links
        ],
        "fixed_body": {
            "link_id": FIXED_BODY_LINK_ID,
            "component_indices": membership["fixed_component_indices"],
        },
        "articulation_root_count": membership["articulation_root_count"],
        "link_membership_basis": membership["membership_basis"],
        "target_candidate_id": task_candidate_ids[0] if task_candidate_ids else None,
        "target_candidate_ids": task_candidate_ids,
        "target_link_id": target_link_id,
        "non_task_candidate_ids": sorted(retained_ids - set(task_candidate_ids)),
        "dropped_candidates": sorted(
            dropped, key=lambda item: item["candidate_id"]
        ),
        "containment_disambiguation": {
            "applied": containment_disambiguation_applied,
            "rejected_containing_link_ids": rejected_containing_link_ids,
        },
        "non_task_joint_mode": non_task_mode,
        "non_task_joint_motion_tolerance": non_task_tolerance,
        "commanded_task_joint_count": commanded_task_joints,
        "candidate_review": rows,
        "claim_boundary": {
            "deterministic_review_is_not_model_accuracy_proof": True,
            "topology_publication_is_not_simready_qualification": True,
            "link_membership_resolved_from_geometry_only": True,
            "physical_equivalence_proven": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def collect_candidate_bounds(
    document: Mapping[str, Any],
    *,
    measure_prim: Callable[[str], tuple[Sequence[float], Sequence[float]] | None],
) -> dict[str, Any]:
    """Measure every candidate's moving prims and declared parent prim.

    ``measure_prim`` returns an ``(aabb_min, aabb_max)`` pair, ``None`` for a
    prim that does not exist, and may raise for a prim the runtime cannot
    measure.  A raising prim becomes a typed per-candidate ``unmeasured``
    record instead of aborting the run: one unmeasurable candidate must never
    destroy the whole lane's evidence (a pxr error previously aborted the run
    before any retention).
    """

    result: dict[str, Any] = {}
    for candidate in document.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        paths = [str(item) for item in candidate.get("moving_part_prims") or []]
        ranges: list[tuple[Sequence[float], Sequence[float]]] = []
        measurement_errors: list[str] = []
        for prim_path in paths:
            try:
                measured = measure_prim(prim_path)
            except Exception as exc:  # noqa: BLE001 - typed per-prim quarantine
                measurement_errors.append(f"{prim_path}:{type(exc).__name__}")
                continue
            if measured is not None:
                ranges.append(measured)
        record: dict[str, Any]
        if ranges:
            record = {
                "status": MEASURED_BOUNDS_STATUS,
                "moving_part_prims": paths,
                "aabb_min": [
                    min(float(low[axis]) for low, _high in ranges)
                    for axis in range(3)
                ],
                "aabb_max": [
                    max(float(high[axis]) for _low, high in ranges)
                    for axis in range(3)
                ],
            }
        else:
            record = {
                "status": UNMEASURED_BOUNDS_STATUS,
                "moving_part_prims": paths,
                "unmeasured_reason": (
                    "prim_measurement_error"
                    if measurement_errors
                    else "no_measurable_prims"
                ),
            }
        if measurement_errors:
            record["measurement_errors"] = measurement_errors
        parent_prim = str(candidate.get("fixed_parent_prim") or "")
        if parent_prim:
            record["fixed_parent_prim"] = parent_prim
            try:
                measured = measure_prim(parent_prim)
                parent_error = None
            except Exception as exc:  # noqa: BLE001 - typed per-prim quarantine
                measured = None
                parent_error = f"{parent_prim}:{type(exc).__name__}"
            if measured is not None:
                record["fixed_parent_bounds"] = {
                    "status": MEASURED_BOUNDS_STATUS,
                    "aabb_min": [float(value) for value in measured[0]],
                    "aabb_max": [float(value) for value in measured[1]],
                }
            else:
                record["fixed_parent_bounds"] = {
                    "status": UNMEASURED_BOUNDS_STATUS,
                    "unmeasured_reason": (
                        "prim_measurement_error"
                        if parent_error
                        else "prim_not_found"
                    ),
                }
                if parent_error:
                    record["fixed_parent_bounds"]["measurement_errors"] = [
                        parent_error
                    ]
        result[candidate_id] = record
    return result


def resolve_joint_agent_output(
    *,
    working_dir: Path,
    role: str,
    relative_glob: str,
    notes: list[str] | None = None,
) -> Path:
    """Resolve one released-code output by role without guessing its filename.

    Symlinks are followed to their real file and deduplicated against direct
    matches; zero-byte and unresolvable matches are skipped with a typed note
    instead of making the role ambiguous.  The released optimizer has emitted
    both a symlink and a zero-byte sibling for the same optimized USD, which
    previously aborted the run and lost its evidence.
    """

    root = working_dir.resolve()
    resolved_matches: dict[Path, str] = {}
    for candidate in sorted(working_dir.glob(relative_glob)):
        relative = candidate.relative_to(working_dir).as_posix()
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            if notes is not None:
                notes.append(f"skipped_unresolvable:{role}:{relative}")
            continue
        if not resolved.is_file():
            if notes is not None:
                notes.append(f"skipped_not_a_file:{role}:{relative}")
            continue
        if resolved.stat().st_size <= 0:
            if notes is not None:
                notes.append(f"skipped_zero_byte:{role}:{relative}")
            continue
        if not resolved.is_relative_to(root):
            if notes is not None:
                notes.append(f"skipped_outside_working_dir:{role}:{relative}")
            continue
        if candidate.is_symlink() and notes is not None:
            notes.append(f"resolved_symlink:{role}:{relative}")
        resolved_matches.setdefault(resolved, relative)
    if len(resolved_matches) != 1:
        raise ValueError(
            f"joint_agent_output_role_not_unique:{role}:observed={len(resolved_matches)}"
        )
    return next(iter(resolved_matches))


__all__ = [
    "ABSORPTION_MAXIMUM_EXTENT_FRACTION",
    "CONTAINMENT_TOLERANCE_FRACTION",
    "FIXED_BODY_LINK_ID",
    "FRAME_COEXTENSIVE_MINIMUM_FRACTION",
    "JointAgentArticulationReviewError",
    "NON_TASK_JOINT_MODES",
    "NON_TASK_JOINT_MODE_EXEMPT",
    "NON_TASK_JOINT_MODE_STRICT",
    "SCHEMA_VERSION",
    "collect_candidate_bounds",
    "resolve_joint_agent_output",
    "resolve_link_membership",
    "review_joint_agent_articulation",
]
