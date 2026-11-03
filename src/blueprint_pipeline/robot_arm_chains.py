"""Each catalog arm's kinematic chain, read from its pinned public robot description, for proper reach checks.

The first fit tier (``website_robot_fit``) judged whether a fixed arm reaches a job by a sphere of its
published reach around the station's base. A reach figure does not say where the shoulder sits, which
space near its own base the arm cannot fold into, what its joint limits allow, or whether the tool can
point down into a pick. The arm's own description does, and Blueprint retains it for every recipe arm.

``read_urdf_arm`` reads a URDF from the arm's base link out to its flange link: each joint's ``origin``
(``xyz``, ``rpy``) and, for a moving joint, its ``axis`` and limits, with the URDF defaults where they are
left out. It reads only the one route between the two links. A mimic joint, a floating or planar joint, a
missing or inverted limit, a zero axis, a link with two parents or no route at all is refused with a typed
code, never guessed. Every flange link the catalog names has its +z out of the flange (the ROS-Industrial
``tool0`` convention); the grippers the pipeline fits to the same bodies confirm it.

``_catalog_data/robot_arm_chains.v1.json`` holds the chain of every fixed arm in the spec catalog whose
recipe has pinned public sources. Each was read from the URDF the pipeline prepared from those sources
(``tests/fixtures/body_kinematics``, rebuilt by its ``regenerate.py``) and names the recipe's source pins
and that URDF's sha256. Only joint frames and limits are kept: no meshes or inertias, and nothing is
fetched at run time. ``python -m blueprint_pipeline.robot_arm_chains --urdf-root <fixtures>`` rewrites the
file from those URDFs. ``load_robot_arm_chains`` fails closed with every reason at once and seals the
catalog with ``catalog_digest``.

``arm_chain`` turns an arm into the ``team_body_kinematics.ArmChain`` whose forward and inverse kinematics
the team gates use, with a tool of a given length straight out of the flange. A solved pose exists in the
publisher's model, with no collision checked against the scene or the arm itself: a CPU bounded estimate,
not a claim about the robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

from .decision_evidence_contracts import DecisionEvidenceContractError, canonical_digest
from .robot_spec_catalog import RECIPE_CATALOG_PATH

SCHEMA_VERSION = "robot_arm_chains.v1"
CLAIM_CEILING = "cpu_bounded_estimate"
CATALOG_PATH = Path(__file__).parent / "_catalog_data" / "robot_arm_chains.v1.json"
MOVING_TYPES = ("revolute", "continuous", "prismatic")
#: The recipe's own source pins, which each arm repeats.
SOURCE_KEYS = ("repository_url", "revision", "entry_path", "license_id")
_FIXED_KEYS = frozenset({"name", "type", "parent", "child", "xyz", "rpy"})
_ROW_KEYS = {"fixed": _FIXED_KEYS, "continuous": _FIXED_KEYS | {"axis"},
             "revolute": _FIXED_KEYS | {"axis", "lower", "upper"}, "prismatic": _FIXED_KEYS | {"axis", "lower", "upper"}}
_ARM_KEYS = frozenset({"recipe_id", "source", "read_from", "base_link", "flange_link", "joints"})
_TOP_KEYS = frozenset({"schema_version", "claim_ceiling", "notice", "arms"})
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class RobotArmChainError(DecisionEvidenceContractError):
    """A chain cannot be read or trusted; ``errors`` lists why."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(_number(item) for item in value)


# --- reading a URDF ------------------------------------------------------------------------------------

def _numbers(text: str | None, default: Sequence[float], code: str) -> list[float]:
    if text is None:
        return [float(value) for value in default]
    try:
        values = [float(value) for value in text.split()]
    except ValueError:
        raise RobotArmChainError([code]) from None
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise RobotArmChainError([code])
    return values


def _link(node: Any, tag: str) -> str:
    end = node.find(tag)
    name = None if end is None else end.get("link")
    if not _text(name):
        raise RobotArmChainError(["robot_arm_urdf_invalid"])
    return name


def _joint_row(node: Any) -> dict[str, Any]:
    name, kind = node.get("name"), node.get("type")
    if kind not in ("fixed", *MOVING_TYPES):
        raise RobotArmChainError([f"robot_arm_urdf_joint_type_unsupported:{name}:{kind}"])
    if node.find("mimic") is not None:
        raise RobotArmChainError([f"robot_arm_urdf_mimic_joint:{name}"])
    origin = node.find("origin")
    row = {"name": name, "type": kind, "parent": _link(node, "parent"), "child": _link(node, "child"),
           **{key: _numbers(None if origin is None else origin.get(key), (0.0, 0.0, 0.0),
                            f"robot_arm_urdf_origin_invalid:{name}") for key in ("xyz", "rpy")}}
    if kind == "fixed":
        return row
    axis = node.find("axis")
    row["axis"] = _numbers(None if axis is None else axis.get("xyz"), (1.0, 0.0, 0.0),
                           f"robot_arm_urdf_axis_invalid:{name}")
    if math.hypot(*row["axis"]) <= 1e-12:
        raise RobotArmChainError([f"robot_arm_urdf_axis_invalid:{name}"])
    if kind == "continuous":
        return row
    limit = node.find("limit")
    try:
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
    except (AttributeError, TypeError, ValueError):
        raise RobotArmChainError([f"robot_arm_urdf_limit_invalid:{name}"]) from None
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise RobotArmChainError([f"robot_arm_urdf_limit_invalid:{name}"])
    return {**row, "lower": lower, "upper": upper}


def read_urdf_arm(path: str | Path, *, base_link: str, flange_link: str) -> list[dict[str, Any]]:
    """The joints from ``base_link`` out to ``flange_link``, in order, or ``RobotArmChainError`` naming why not."""
    try:
        root = ET.parse(Path(path)).getroot()
    except (ET.ParseError, DefusedXmlException, OSError) as exc:
        raise RobotArmChainError(["robot_arm_urdf_invalid"]) from exc
    if root.tag != "robot":
        raise RobotArmChainError(["robot_arm_urdf_invalid"])
    links = {node.get("name") for node in root.findall("link")}
    for name in (base_link, flange_link):
        if name not in links:
            raise RobotArmChainError([f"robot_arm_urdf_link_missing:{name}"])
    above: dict[str, Any] = {}
    for node in root.findall("joint"):
        child = _link(node, "child")
        if child in above:
            raise RobotArmChainError([f"robot_arm_urdf_link_has_two_parents:{child}"])
        above[child] = node
    route, link = [], flange_link
    while link != base_link:
        # More steps than joints means a loop; no joint above means the base is not above the flange.
        if link not in above or len(route) > len(above):
            raise RobotArmChainError([f"robot_arm_urdf_flange_not_below_base:{flange_link}"])
        route.append(above[link])
        link = _link(above[link], "parent")
    rows = [_joint_row(node) for node in reversed(route)]
    if not any(row["type"] in MOVING_TYPES for row in rows):
        raise RobotArmChainError(["robot_arm_urdf_no_movable_joint"])
    return rows


# --- the catalog ---------------------------------------------------------------------------------------

def _recipe_sources() -> dict[str, dict[str, Any]]:
    value = json.loads(RECIPE_CATALOG_PATH.read_text(encoding="utf-8"))
    return {str(row["embodiment_id"]): dict(row.get("source") or {}) for row in value["recipes"]}


def _row_valid(row: Any) -> bool:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS.get(row.get("type"), frozenset()):
        return False
    if not all(_text(row[key]) for key in ("name", "parent", "child")) or not all(
            _vector(row[key]) for key in ("xyz", "rpy")):
        return False
    if row["type"] == "fixed":
        return True
    if not _vector(row["axis"]) or math.hypot(*row["axis"]) <= 1e-12:
        return False
    return row["type"] == "continuous" or (_number(row["lower"]) and _number(row["upper"])
                                           and row["lower"] <= row["upper"])


def _arm_errors(key: str, arm: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errors = []
    if set(arm) != _ARM_KEYS:
        errors.append(f"robot_arm_chains_arm_fields_invalid:{key}")
    if key not in sources:
        errors.append(f"robot_arm_chains_recipe_unknown:{key}")
    elif arm.get("source") != {name: sources[key].get(name) for name in SOURCE_KEYS}:
        errors.append(f"robot_arm_chains_source_mismatch:{key}")
    read = arm.get("read_from")
    if not (isinstance(read, dict) and set(read) == {"body", "urdf_sha256"} and read["body"] in sources
            and isinstance(read["urdf_sha256"], str) and _SHA256.fullmatch(read["urdf_sha256"])):
        errors.append(f"robot_arm_chains_read_from_invalid:{key}")
    base, flange, joints = arm.get("base_link"), arm.get("flange_link"), arm.get("joints")
    if not _text(base) or not _text(flange) or not isinstance(joints, list) or not joints:
        return [*errors, f"robot_arm_chains_route_broken:{key}"]
    link, broken = base, False
    for index, row in enumerate(joints):
        if not _row_valid(row):
            errors.append(f"robot_arm_chains_joint_invalid:{key}:{index}")
        broken = broken or not isinstance(row, dict) or row.get("parent") != link
        link = row.get("child") if isinstance(row, dict) else None
    if broken or link != flange:
        errors.append(f"robot_arm_chains_route_broken:{key}")
    if not any(isinstance(row, dict) and row.get("type") in MOVING_TYPES for row in joints):
        errors.append(f"robot_arm_chains_no_movable_joint:{key}")
    return errors


def validate_robot_arm_chains(value: Mapping[str, Any]) -> dict[str, Any]:
    """The catalog, sealed with ``catalog_digest``; raise ``RobotArmChainError`` with every reason."""
    try:
        catalog = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RobotArmChainError(["robot_arm_chains_not_json"]) from exc
    if not isinstance(catalog, dict):
        raise RobotArmChainError(["robot_arm_chains_not_json"])
    catalog.pop("catalog_digest", None)
    errors: list[str] = []
    if catalog.get("schema_version") != SCHEMA_VERSION:
        errors.append("robot_arm_chains_schema_invalid")
    if catalog.get("claim_ceiling") != CLAIM_CEILING or set(catalog) != _TOP_KEYS or not _text(catalog.get("notice")):
        errors.append("robot_arm_chains_fields_invalid")
    arms = catalog.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("robot_arm_chains_arms_missing")
        arms = []
    sources, seen = _recipe_sources(), set()
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict) or not _text(arm.get("recipe_id")):
            errors.append(f"robot_arm_chains_arm_invalid:{index}")
            continue
        key = arm["recipe_id"]
        if key in seen:
            errors.append(f"robot_arm_chains_duplicate_recipe:{key}")
        seen.add(key)
        errors += _arm_errors(key, arm, sources)
    if errors:
        raise RobotArmChainError(errors)
    catalog["catalog_digest"] = canonical_digest(catalog, digest_field="catalog_digest")
    return catalog


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            # A repeated key would silently replace a joint or a pin.
            raise RobotArmChainError([f"robot_arm_chains_duplicate_json_key:{key}"])
        value[key] = item
    return value


def load_robot_arm_chains(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate a chain catalog (the shipped ``CATALOG_PATH`` by default)."""
    source = Path(path) if path is not None else CATALOG_PATH
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except RobotArmChainError:
        raise
    except ValueError as exc:
        raise RobotArmChainError(["robot_arm_chains_not_json"]) from exc
    return validate_robot_arm_chains(value)


def refresh_robot_arm_chains(value: Mapping[str, Any], *, urdf_root: str | Path) -> dict[str, Any]:
    """``value`` with each arm's source pins, URDF digest and joints read again from
    ``<urdf_root>/<read_from.body>/robot.urdf``; the recipe, body and links that name an arm are kept."""
    sources, arms = _recipe_sources(), []
    for arm in value["arms"]:
        body, key = arm["read_from"]["body"], arm["recipe_id"]
        urdf = Path(urdf_root) / body / "robot.urdf"
        arms.append({"recipe_id": key, "source": {name: sources[key][name] for name in SOURCE_KEYS},
                     "read_from": {"body": body, "urdf_sha256": "sha256:" + hashlib.sha256(urdf.read_bytes()).hexdigest()},
                     "base_link": arm["base_link"], "flange_link": arm["flange_link"],
                     "joints": read_urdf_arm(urdf, base_link=arm["base_link"], flange_link=arm["flange_link"])})
    return {**{key: item for key, item in value.items() if key != "catalog_digest"}, "arms": arms}


def _dumps(value: Mapping[str, Any]) -> str:
    """The catalog as written: one joint per line, so a diff names the joint that changed."""
    arms = []
    for arm in value["arms"]:
        fields = [f"   {json.dumps(key)}: {json.dumps(item)}" for key, item in arm.items() if key != "joints"]
        joints = ",\n".join(f"    {json.dumps(row)}" for row in arm["joints"])
        arms.append("  {\n" + ",\n".join([*fields, f'   "joints": [\n{joints}\n   ]']) + "\n  }")
    fields = [f" {json.dumps(key)}: {json.dumps(item)}" for key, item in value.items() if key != "arms"]
    return "{\n" + ",\n".join([*fields, ' "arms": [\n' + ",\n".join(arms) + "\n ]"]) + "\n}\n"


# --- the solver's chain --------------------------------------------------------------------------------

def _origin(row: Mapping[str, Any]) -> np.ndarray:
    """A URDF origin: translate by ``xyz``, then rotate by fixed-axis roll, pitch and yaw."""
    roll, pitch, yaw = row["rpy"]
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    result = np.eye(4)
    result[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]]
    result[:3, 3] = row["xyz"]
    return result


def _z_onto(axis: Sequence[float]) -> np.ndarray:
    """A rotation turning +z onto the unit ``axis``, so a joint turns (or slides) about its frame's z."""
    unit = np.asarray(axis, dtype=float) / math.hypot(*axis)
    cos, cross = float(unit[2]), np.array([-unit[1], unit[0], 0.0])
    sin = float(np.linalg.norm(cross))
    result = np.eye(4)
    if sin < 1e-12:
        result[:3, :3] = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
        return result
    x, y, z = cross / sin
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    result[:3, :3] = np.eye(3) + sin * skew + (1.0 - cos) * skew @ skew
    return result


def arm_chain(arm: Mapping[str, Any], *, tool_length_m: float) -> Any:
    """The validated ``arm``'s ``team_body_kinematics.ArmChain``, its base frame the base link's, with a
    tool ``tool_length_m`` long straight out of the flange (its TCP; the flange's +z is out of the flange)."""
    from .team_body_kinematics import IK_MAX_PRISMATIC_STEP_M, IK_MAX_REVOLUTE_STEP_RAD, ArmChain

    if not _number(tool_length_m) or tool_length_m < 0:
        raise RobotArmChainError(["robot_arm_tool_length_invalid"])
    # Walk the joints keeping ``current``: the latest frame (the base, or the last joint's moving frame).
    current, before, names, kinds, lower, upper = np.eye(4), [], [], [], [], []
    for row in arm["joints"]:
        if row["type"] == "fixed":
            current = current @ _origin(row)
            continue
        onto = _z_onto(row["axis"])
        before.append(current @ _origin(row) @ onto)
        # The child link is the joint frame turned back off its axis.
        current = onto.T.copy()
        names.append(row["name"])
        kinds.append("prismatic" if row["type"] == "prismatic" else "revolute")
        lower.append(row.get("lower", -math.inf))
        upper.append(row.get("upper", math.inf))
    count = len(names)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    # Start mid-range; an unlimited (continuous) joint starts at zero.
    reset, limited = np.zeros(count), np.isfinite(low) & np.isfinite(high)
    reset[limited] = (low[limited] + high[limited]) / 2.0
    prismatic = np.asarray([kind == "prismatic" for kind in kinds])
    tool_from_tcp = np.eye(4)
    tool_from_tcp[2, 3] = float(tool_length_m)
    frames = np.asarray(before)
    # From the first moving joint, no point of the arm is further than every link offset, the TCP and
    # every slide's travel laid end to end.
    travel = float(np.maximum(np.abs(low), np.abs(high))[prismatic].sum())
    reach = float(np.linalg.norm(frames[1:, :3, 3], axis=1).sum() + np.linalg.norm((current @ tool_from_tcp)[:3, 3])
                  + travel)
    return ArmChain(name=str(arm["recipe_id"]), robot_usd="urdf:" + str(arm["read_from"]["body"]),
                    robot_usd_sha256=arm["read_from"]["urdf_sha256"], base_link=arm["base_link"],
                    tool_link=arm["flange_link"], tool_path=arm["flange_link"], joint_names=tuple(names),
                    joint_kinds=tuple(kinds), lower=low, upper=high, reset=reset, authored=np.zeros(count), bodies=(), tool_from_tcp=tool_from_tcp, root_from_base=np.eye(4),
                    reach_bound_m=reach, notes=("model_format:urdf", "reset_unavailable_mid_range_used"),
                    order=np.arange(count), sign=np.ones(count), prismatic=prismatic, before=frames,
                    joint_from_tool=current,
                    max_step=np.where(prismatic, IK_MAX_PRISMATIC_STEP_M, IK_MAX_REVOLUTE_STEP_RAD))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite the arm chain catalog from the prepared robot URDFs.")
    parser.add_argument("--catalog", default=str(CATALOG_PATH))
    parser.add_argument("--urdf-root", required=True,
                        help="the folder holding <body>/robot.urdf, e.g. tests/fixtures/body_kinematics")
    args = parser.parse_args(argv)
    path = Path(args.catalog)
    value = refresh_robot_arm_chains(json.loads(path.read_text(encoding="utf-8")), urdf_root=args.urdf_root)
    validate_robot_arm_chains(value)
    path.write_text(_dumps(value), encoding="utf-8")
    return 0


__all__ = ["CATALOG_PATH", "CLAIM_CEILING", "SCHEMA_VERSION", "RobotArmChainError", "arm_chain",
           "load_robot_arm_chains", "main", "read_urdf_arm", "refresh_robot_arm_chains", "validate_robot_arm_chains"]


if __name__ == "__main__":
    raise SystemExit(main())
