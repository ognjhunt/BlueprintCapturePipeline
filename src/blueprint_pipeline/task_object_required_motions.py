"""ADP-030/day 28: task-required motions, independent of object/site names."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


def assembly_mounting(value, *, source_frame_ids=None):
    """A capture-reviewed base constraint is a candidate, not measured mounting."""
    if value is None:
        return None
    if (not isinstance(value, Mapping) or set(value) != {"mode", "source_frame_ids", "reason"}
            or not isinstance(value["mode"], str) or value["mode"] not in {"fixed_in_scene", "free", "unknown"}
            or not isinstance(value["reason"], str) or not value["reason"].strip()
            or not isinstance(value["source_frame_ids"], list)
            or any(not isinstance(i, str) or not i for i in value["source_frame_ids"])
            or len(set(value["source_frame_ids"])) != len(value["source_frame_ids"])
            or value["mode"] != "unknown" and not value["source_frame_ids"]
            or source_frame_ids is not None and not set(value["source_frame_ids"]) <= set(source_frame_ids)):
        raise ValueError("task_object_assembly_mounting_invalid")
    return {**value, "source_frame_ids": sorted(value["source_frame_ids"])}


def merge_assembly_mounting(rows):
    """Conflicting visible base constraints stay unknown; object names do not vote."""
    rows = [assembly_mounting(row) for row in rows if row is not None]
    if not rows:
        return None
    modes = {row["mode"] for row in rows} - {"unknown"}
    mode = next(iter(modes)) if len(modes) == 1 else "unknown"
    evidence = [row for row in rows if row["mode"] == mode] if mode != "unknown" else rows
    return {"mode": mode, "source_frame_ids": sorted({i for row in evidence for i in row["source_frame_ids"]}),
            "reason": ("Conflicting capture reviews; mounting remains unknown." if len(modes) > 1 else
                       " ".join(dict.fromkeys(row["reason"] for row in evidence)))}


def required_motions(value, *, known_parts=None, source_frame_ids=None):
    """Requirements describe intent and visible parts, never measured mechanics."""
    if not isinstance(value, list):
        raise ValueError("task_object_required_motions_invalid")
    rows, names = [], set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "motion_id", "part_ids", "joint_type", "task_actions", "source_frame_ids"
        }:
            raise ValueError("task_object_required_motions_invalid")
        name = row["motion_id"]
        if (not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
                or name in names or not isinstance(row["joint_type"], str)
                or row["joint_type"] not in {"revolute", "prismatic"}):
            raise ValueError("task_object_required_motions_invalid")
        for field in ("part_ids", "task_actions", "source_frame_ids"):
            items = row[field]
            if (not isinstance(items, list) or (field != "source_frame_ids" and not items)
                    or any(not isinstance(item, str) or not item.strip() for item in items)
                    or len(set(items)) != len(items)):
                raise ValueError("task_object_required_motions_invalid")
        parts = set(row["part_ids"])
        if (known_parts is not None and not parts <= set(known_parts)
                or source_frame_ids is not None and not set(row["source_frame_ids"]) <= set(source_frame_ids)):
            raise ValueError("task_object_required_motions_invalid")
        names.add(name)
        rows.append({**row, **{field: sorted(row[field]) for field in
                              ("part_ids", "task_actions", "source_frame_ids")}})
    return sorted(rows, key=lambda row: row["motion_id"])


def merge_required_motions(previous, additions):
    """Batch-local views may reveal more members/states of the same motion."""
    merged = {row["motion_id"]: dict(row) for row in previous}
    for row in additions:
        prior = merged.get(row["motion_id"])
        if prior is None:
            merged[row["motion_id"]] = dict(row)
            continue
        if prior["joint_type"] != row["joint_type"]:
            raise ValueError("task_object_required_motion_type_changed")
        merged[row["motion_id"]] = {**prior, **{field: sorted(set(prior[field]) | set(row[field]))
            for field in ("part_ids", "task_actions", "source_frame_ids")}}
    return required_motions(list(merged.values()))


def motion_graph_readback(requirements: Sequence[Mapping], graph: Mapping, *, part_links=None):
    """Require distinct, usable exported joints for independent requested motions.

    Fixed child features inherit their carrier's motion. A part's presence in
    a mesh, or two named parts on one sliding body, cannot satisfy two motions.
    This checks capability; the existing task scorer still owns task success.
    """
    requirements = required_motions(list(requirements))
    links = {row["link_id"] for row in graph["links"]}
    incoming = {row["child_link_id"]: row for row in graph["joints"]}
    joints = {row["joint_id"]: row for row in graph["joints"]}
    used, rows = set(), []
    for requirement in requirements:
        common, resolved = None, {}
        for part in requirement["part_ids"]:
            link = (part_links or {}).get(part, part)
            if link not in links:
                raise ValueError("artisan_required_motion_part_missing:" + part)
            resolved[part] = link
            candidates, seen = set(), set()
            while link in incoming:
                if link in seen:
                    raise ValueError("artisan_required_motion_graph_cycle")
                seen.add(link)
                joint = incoming[link]
                bounds = joint["limits"]
                if (joint["joint_type"] == requirement["joint_type"] and joint["role"] != "locked"
                        and len(bounds) == 2 and all(math.isfinite(v) for v in bounds)
                        and bounds[0] < bounds[1]):
                    candidates.add(joint["joint_id"])
                link = joint["parent_link_id"]
            common = candidates if common is None else common & candidates
        if not common:
            raise ValueError("artisan_required_task_motion_missing:" + requirement["motion_id"])
        if len(common) != 1:
            raise ValueError("artisan_required_task_motion_ambiguous:" + requirement["motion_id"])
        joint_id = next(iter(common))
        if joint_id in used:
            raise ValueError("artisan_independent_task_motions_share_joint:" + joint_id)
        used.add(joint_id)
        rows.append({"motion_id": requirement["motion_id"], "part_links": resolved,
                     "joint_id": joint_id, "joint_type": joints[joint_id]["joint_type"],
                     "limits": joints[joint_id]["limits"], "task_actions": requirement["task_actions"]})
    return {"schema_version": "task_object_required_motion_readback.v1", "motions": rows,
            "requirements_digest": canonical_digest({"required_motions": requirements}),
            "task_success_proven": False, "physics_authority_granted": False}
