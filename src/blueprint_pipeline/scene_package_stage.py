"""One composed stage for a job's simulation, written from the scene plan, for files that leave Blueprint.

The compiled scene's USD files are separate assets; where each sits lives only in the scene plan
(``native_task_arena_scene_plan.v1.json``, written by ``native_task_arena_packet``), and the runtime places each
row's asset at its ``pose_world`` (``native_task_arena_runtime``: the pose describes the USD namespace). The plan
is a record and never leaves. So both the simulation package for deployment companies
(``website_company_simulation``) and the offline job packet (``job_packet_export``) write ``scene.usda`` with this
module: one stage that references each shipped asset by a relative path, under a neutral prim name for its role,
at the plan's position, orientation and scale, with the plan's visibility, in metres with the plan's up axis.

Nothing from the plan is copied as text: no id, digest, path or name. A row that cannot be placed honestly (an
unknown role, a missing or malformed pose, an asset that is not shipped) refuses the stage, naming the role. No
pose is ever guessed.

``neutral_rule`` projects the plan's success criteria and the few neutral facts a company needs to reproduce the
run (physics and control rates, the episode length, the object's start pose and size) into the small success-rule
file: numbers, true or false, and short lowercase mode words only.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

#: The neutral prim name for each role the scene plan uses.
ROLE_PRIMS = {"scene_appearance": "Room", "scene_collision": "RoomCollision", "task_object": "Object",
              "task_support": "Support", "replacement": "Replacement"}
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_MODE_WORD = re.compile(r"[a-z][a-z_]{0,40}")
#: Criteria fields that name things (ids, digests, paths, labels) or Blueprint's own checks; never shipped.
_DROPPED = frozenset({"controls", "visible_label", "schema_version", "target_digest", "support_prim_path",
                      "support_source_instance_id"})
_DROPPED_ENDINGS = ("_id", "_ids", "_digest", "_path", "_paths", "_label", "_uri", "_name")


class StageRefused(ValueError):
    """A row of the plan cannot be placed honestly; ``role`` names it."""

    def __init__(self, reason: str, role: str = ""):
        super().__init__(f"{reason}:{role}" if role else reason)
        self.reason, self.role = reason, role


def _finite(value: Any, length: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
                    for item in value))


def up_axis(plan: Mapping[str, Any]) -> str:
    """The stage's up axis from the plan's lift direction; refused when the plan does not say Z or Y."""
    lift = ((plan.get("task_spec") or {}).get("interaction_affordance") or {}).get("lift_unit_world")
    if _finite(lift, 3) and [float(item) for item in lift] == [0.0, 0.0, 1.0]:
        return "Z"
    if _finite(lift, 3) and [float(item) for item in lift] == [0.0, 1.0, 0.0]:
        return "Y"
    raise StageRefused("scene_stage_up_axis_not_stated")


def placements(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every object row of the plan as a placement; ``StageRefused`` naming the role of any row that cannot be."""
    rows = plan.get("objects")
    if not isinstance(rows, list) or not rows:
        raise StageRefused("scene_stage_no_objects")
    found, seen = [], set()
    for row in rows:
        role = str(row.get("semantic_role") or "") if isinstance(row, Mapping) else ""
        prim = ROLE_PRIMS.get(role)
        if prim is None:
            raise StageRefused("scene_stage_role_unknown", role if _MODE_WORD.fullmatch(role) else "unnamed")
        if prim in seen:
            raise StageRefused("scene_stage_role_repeated", role)
        seen.add(prim)
        pose = row.get("pose_world")
        position = pose.get("position_world_m") if isinstance(pose, Mapping) else None
        orientation = pose.get("orientation_xyzw") if isinstance(pose, Mapping) else None
        if not _finite(position, 3) or not _finite(orientation, 4) or not math.isclose(
                sum(item * item for item in orientation), 1.0, abs_tol=1e-3):
            raise StageRefused("scene_stage_pose_missing", role)
        scale = row.get("scale")
        if scale is not None and not (_finite(scale, 3) and all(item > 0 for item in scale)):
            raise StageRefused("scene_stage_scale_invalid", role)
        relative = PurePosixPath(str(row.get("usd_path") or ""))
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or relative.suffix.lower() not in USD_SUFFIXES):
            raise StageRefused("scene_stage_asset_not_shipped", role)
        if not isinstance(row.get("visible", True), bool):
            raise StageRefused("scene_stage_visibility_invalid", role)
        found.append({"role": role, "prim": prim, "usd_path": relative.as_posix(),
                      "position": [float(item) for item in position],
                      "orientation_xyzw": [float(item) for item in orientation],
                      "scale": [float(item) for item in scale] if scale is not None else None,
                      "visible": row.get("visible", True) is not False})
    return found


def _target_prim(path: Path, role: str) -> str:
    """'' when the layer names a default prim, else its first root prim; refused when it has none."""
    from pxr import Sdf
    try:
        layer = Sdf.Layer.OpenAsAnonymous(str(path))
    except Exception:  # noqa: BLE001 - any failure to open is a refusal
        layer = None
    if layer is None:
        raise StageRefused("scene_stage_asset_unreadable", role)
    if layer.defaultPrim:
        return ""
    roots = [prim.path.pathString for prim in layer.rootPrims]
    if not roots:
        raise StageRefused("scene_stage_asset_unreadable", role)
    return roots[0]


def _number(value: float) -> str:
    return repr(float(value))


def write_stage(stage_path: Path, rows: Sequence[Mapping[str, Any]], *, axis: str) -> str:
    """Write ``scene.usda`` at ``stage_path``; each row's ``file`` is the shipped asset, relative to the stage.

    The text is written here, not by ``pxr``, so the same rows give the same bytes. Returns the text.
    """
    stage_path = Path(stage_path)
    lines = ["#usda 1.0", "(", '    defaultPrim = "World"', "    metersPerUnit = 1", f'    upAxis = "{axis}"', ")", "",
             'def Xform "World"', "{"]
    for row in rows:
        relative = PurePosixPath(str(row["file"]))
        shipped = stage_path.parent.joinpath(*relative.parts)
        if (relative.is_absolute() or ".." in relative.parts or not shipped.is_file()
                or not shipped.resolve().is_relative_to(stage_path.parent.resolve())):
            raise StageRefused("scene_stage_asset_not_shipped", row["role"])
        prim = _target_prim(shipped, row["role"])
        x, y, z = row["position"]
        qx, qy, qz, qw = row["orientation_xyzw"]
        order = ['"xformOp:translate"', '"xformOp:orient"']
        lines += [f'    def Xform "{row["prim"]}" (',
                  f"        prepend references = @./{relative.as_posix()}@{f'<{prim}>' if prim else ''}", "    )",
                  "    {"]
        if not row["visible"]:
            lines.append('        token visibility = "invisible"')
        lines += [f"        quatd xformOp:orient = ({_number(qw)}, {_number(qx)}, {_number(qy)}, {_number(qz)})"]
        if row.get("scale") is not None:
            sx, sy, sz = row["scale"]
            lines.append(f"        double3 xformOp:scale = ({_number(sx)}, {_number(sy)}, {_number(sz)})")
            order.append('"xformOp:scale"')
        lines += [f"        double3 xformOp:translate = ({_number(x)}, {_number(y)}, {_number(z)})",
                  f"        uniform token[] xformOpOrder = [{', '.join(order)}]", "    }"]
    lines += ["}", ""]
    text = "\n".join(lines)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(text, encoding="utf-8")
    return text


def _neutral(value: Any) -> Any:
    """Numbers, true or false, and short mode words; anything that could name something is dropped."""
    if isinstance(value, bool) or (isinstance(value, (int, float)) and math.isfinite(value)):
        return value
    if isinstance(value, str):
        return value if _MODE_WORD.fullmatch(value) else None
    if isinstance(value, list):
        items = [_neutral(item) for item in value]
        return items if all(item is not None for item in items) else None
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or key in _DROPPED or key.endswith(_DROPPED_ENDINGS):
                continue
            kept = _neutral(item)
            if kept is not None and kept != {}:
                out[key] = kept
        return out
    return None


def neutral_rule(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    """The success-rule file: the criteria and the neutral facts to reproduce the run, or None without criteria."""
    criteria = ((plan.get("task_spec") or {}).get("task_success_contract") or {}).get("criteria")
    if not isinstance(criteria, Mapping):
        return None
    rule: dict[str, Any] = {"units": "metres, radians, seconds", "up_axis": up_axis(plan),
                            "criteria": _neutral(criteria)}
    cadence = plan.get("cadence") if isinstance(plan.get("cadence"), Mapping) else {}
    timing = {key: cadence[key] for key in ("physics_frequency_hz", "control_frequency_hz", "episode_length_seconds")
              if isinstance(cadence.get(key), (int, float)) and not isinstance(cadence.get(key), bool)
              and math.isfinite(cadence[key])}
    if timing:
        rule["timing"] = timing
    for row in plan.get("objects") or []:
        if isinstance(row, Mapping) and row.get("semantic_role") == "task_object":
            pose = row.get("pose_world") or {}
            if _finite(pose.get("position_world_m"), 3) and _finite(pose.get("orientation_xyzw"), 4):
                rule["object_start"] = {"position_m": list(pose["position_world_m"]),
                                        "orientation_xyzw": list(pose["orientation_xyzw"])}
    extent = (plan.get("task_object_observability") or {}).get("task_object_extent_m")
    if _finite(extent, 3):
        rule["object_size_m"] = list(extent)
    return rule


__all__ = ["ROLE_PRIMS", "StageRefused", "neutral_rule", "placements", "up_axis", "write_stage"]
