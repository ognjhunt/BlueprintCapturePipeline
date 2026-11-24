"""One composed stage for a job's simulation, written from the scene plan, for files that leave Blueprint.

The compiled scene's USD files are separate assets; where each sits lives only in the scene plan
(``native_task_arena_scene_plan.v1.json``, written by ``native_task_arena_packet``), and the runtime places each
row's asset at its ``pose_world`` (``native_task_arena_runtime``: the pose describes the USD namespace). The plan
is a record and never leaves. So the simulation package for deployment companies
(``website_company_simulation``) writes ``scene.usda`` with this
module: one stage with a neutral prim for each row (``/World/Room``, ``/World/Object``, ...; a role this module
does not know becomes ``/World/Part_01`` and so on, a repeated role ``/World/Object_02``), carrying the plan's
position, orientation, scale and visibility, in metres with the plan's up axis. The up axis is the plan's lift
direction (a rigid pick-and-place plan states it); a plan that states no lift direction (an articulated open/close
plan) is read in the frame its room files declare (their authored ``upAxis``, which must agree). Neither stated:
refused. No axis is ever assumed.

Each asset is referenced on a child prim, ``<role prim>/Asset``, so the asset keeps its own root transform and the
plan's pose is applied on top of it. That is how Blueprint's own checks compose them: the appearance alignment
gate (``native_task_appearance_frame_alignment``) applies the spawn pose to the asset's own composed transform
(``spawn @ layer``), and the runtime gives the pose to the spawner as the asset's placement in the scene.

Nothing from the plan is copied as text: no id, digest, path or name. A row that cannot be placed honestly (a
missing or malformed pose, an asset that is not shipped) refuses the stage, naming the role. No pose is ever
guessed.

``neutral_rule`` projects the plan's success criteria and the few neutral facts a company needs to reproduce the
run (physics and control rates, the episode length, the object's start pose and size, and the object's estimated
mass and friction when the job has them) into the small success-rule file. The criteria pass through an
allow-list: only keys listed here, holding numbers, true or false, or a word from a closed set, survive; anything
else is left out and counted. An articulated rule's joints are the one exception, read from the shipped object file
rather than the plan: the target joint ships as its prim path in scene.usda, its kind (prismatic or revolute) and the
unit of its numbers, and each locked joint as its prim path and kind (``joint_prims``), never by Blueprint's joint id.
``units`` names the unit of every number the file holds (``rule_units``), and what a window's samples count.
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
#: The success criteria's own keys (rigid_task_success_contract.v1 and what the scorer reads). Nothing else ships.
CRITERIA_KEYS = frozenset({
    "destination_containment", "gripper_state", "motion", "orientation", "retreat", "safety", "settling", "support",
    "surface_target", "temporal_invariants", "terminal_task_contact", "no_drop", "opening", "hold", "mode",
    "position_bounds_world_m", "minimum", "maximum", "threshold_m", "minimum_lift_m", "minimum_translation_m",
    "minimum_translation", "movement_epsilon_m", "reference_xyzw", "tolerance_rad", "minimum_clearance_m",
    "withdrawal_unit_destination_frame", "orientation_tolerance_rad", "position_tolerance_m", "window_samples",
    "contact_mode", "height_interval_m", "height_mode", "maximum_angular_speed_rad_s", "maximum_linear_speed_m_s",
    "maximum_tilt_rad", "non_colliding", "radius_m", "shape", "stable_seconds", "surface_position_world_m",
    "containment_excursions", "workspace_excursions", "forbidden_contact_classes", "maximum_regrasps",
    "maximum_retries", "maximum_task_contact_force_n", "minimum_fall_m", "minimum_opening_m", "minimum_angle_rad",
    "minimum_hold_seconds", "placement_relation", "interior_bounds_world_m", "success_rule",
    # An articulated open/close goal (``articulated_task_success_contract``): the target joint's success interval,
    # hard limits and reset position (metres for a slide, radians for a hinge), the settled speed and the motion
    # threshold. Plain numbers; the joint ids beside them are Blueprint's own names, so the joint blocks
    # (``JOINT_BLOCKS``) ship each joint as its prim path in scene.usda and its kind instead, and ``units`` names the
    # unit of every number of the target joint's own (``rule_units``).
    "success_interval", "joint_hard_limits", "reset_position", "maximum_settled_target_speed", "movement_epsilon"})
#: The articulated criteria's joint blocks: ``target_joint`` (``joint_id``) and ``locked_joints`` (``joint_ids``,
#: ``mode``, ``motion_tolerance``). A joint ships only as read from the job's object file (``joint_prims``).
JOINT_BLOCKS = ("target_joint", "locked_joints")
#: A joint's kind, as its USD type says, and the unit of its own coordinate and of its speed.
JOINT_KINDS = {"prismatic": ("metres", "metres per second"), "revolute": ("radians", "radians per second"),
               "fixed": (None, None)}
#: The target joint's own numbers: in its unit (``JOINT_KINDS``), and its speed in that unit per second.
TARGET_JOINT_FIELDS = (("opening", "success_interval"), ("opening", "joint_hard_limits"),
                       ("opening", "reset_position"), ("motion", "movement_epsilon"))
TARGET_JOINT_SPEED_FIELDS = (("hold", "maximum_settled_target_speed"),)
LOCKED_JOINT_UNIT = ("each locked joint's own unit: metres for a prismatic joint, radians for a revolute one; a fixed "
                     "joint does not move")
#: When the object's file names no target joint (it cannot be read), its numbers' unit is said as it is.
UNKNOWN_JOINT_UNIT = "the target joint's own unit: metres for a prismatic joint, radians for a revolute one"
UNKNOWN_JOINT_SPEED = ("the target joint's own unit a second: metres per second for a prismatic joint, radians per "
                       "second for a revolute one")
#: The unit each key suffix means, longest first; ``units`` lists those the shipped rule uses (``rule_units``).
SUFFIX_UNITS = (("_m_s", "metres per second"), ("_rad_s", "radians per second"), ("_seconds", "seconds"),
                ("_hz", "hertz"), ("_kg", "kilograms"), ("_rad", "radians"), ("_m", "metres"), ("_n", "newtons"))
#: A settling or hold window is counted in the run's control steps (``native_task_arena_scene_plan._cadence``: the
#: episode is its action steps and its settle window at the control rate).
WINDOW_SAMPLES = "control steps"
_PRIM_PATH = re.compile(r"/World/[A-Za-z_][A-Za-z0-9_]*/Asset(?:/[A-Za-z_][A-Za-z0-9_]*)*")
#: The words a criterion's value may be, each read from the scorer, not guessed: the modes
#: ``adp_rigid_task_scoring`` compares (``required``, ``ignored``, ``released``, ``cleared``, ``closed_at_most``,
#: and ``forbidden`` for excursions), the target's shape, relation and rule (``task_evaluation_surface_target``:
#: ``flat_green_disc``, ``on``, ``inside``, ``CENTRE_RULE``) and the contact classes the episode telemetry
#: measures (``native_rigid_episode_telemetry.CONTACT_CHANNELS``). Any other text is left out.
MODE_WORDS = frozenset({
    "required", "ignored", "released", "cleared", "closed_at_most", "forbidden", "flat_green_disc", "on",
    "inside", "subject_centre_within_radius", "robot_background", "robot_object", "object_background",
    "destination_background"})


class StageRefused(ValueError):
    """A row of the plan cannot be placed honestly; ``role`` names it."""

    def __init__(self, reason: str, role: str = ""):
        super().__init__(f"{reason}:{role}" if role else reason)
        self.reason, self.role = reason, role


def _finite(value: Any, length: int) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
                    for item in value))


#: The rows whose files are the room: their own frame is the scene's world frame.
WORLD_FRAME_ROLES = frozenset({"scene_collision", "scene_appearance"})


def up_axis(plan: Mapping[str, Any], *, packet: Path | None = None) -> str:
    """The stage's up axis: the plan's lift direction, else what the room's files declare; refused when neither says.

    A plan with a lift direction must say Z or Y. A plan with none (an articulated open/close plan) is read from
    ``packet``, the directory its rows' files are in: each room file (``WORLD_FRAME_ROLES``) that authors ``upAxis``
    must say the same Z or Y, and at least one must author it. Nothing stated, a disagreement or another axis:
    refused, with the same reason.
    """
    affordance = (plan.get("task_spec") or {}).get("interaction_affordance") or {}
    lift = affordance.get("lift_unit_world")
    if _finite(lift, 3) and [float(item) for item in lift] == [0.0, 0.0, 1.0]:
        return "Z"
    if _finite(lift, 3) and [float(item) for item in lift] == [0.0, 1.0, 0.0]:
        return "Y"
    if "lift_unit_world" not in affordance and packet is not None:
        declared = _room_up_axes(plan, Path(packet))
        if len(declared) == 1 and declared <= {"Z", "Y"}:
            return next(iter(declared))
    raise StageRefused("scene_stage_up_axis_not_stated")


def _room_up_axes(plan: Mapping[str, Any], packet: Path) -> set[str]:
    """The ``upAxis`` each room file of the plan authors (unauthored files say nothing; an unreadable one says '?')."""
    from pxr import Sdf

    declared: set[str] = set()
    for row in plan.get("objects") or []:
        if not isinstance(row, Mapping) or row.get("semantic_role") not in WORLD_FRAME_ROLES:
            continue
        relative = PurePosixPath(str(row.get("usd_path") or ""))
        path = packet.joinpath(*relative.parts)
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.is_file() or not path.resolve().is_relative_to(packet.resolve())):
            declared.add("?")
            continue
        try:
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            root = layer.pseudoRoot
            if root.HasInfo("upAxis"):
                declared.add(str(root.GetInfo("upAxis")))
        except Exception:  # noqa: BLE001 - a file that cannot be read states nothing usable
            declared.add("?")
    return declared


def placements(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every object row of the plan as a placement; ``StageRefused`` naming the role of any row that cannot be."""
    rows = plan.get("objects")
    if not isinstance(rows, list) or not rows:
        raise StageRefused("scene_stage_no_objects")
    found: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    parts = 0
    for row in rows:
        role = str(row.get("semantic_role") or "") if isinstance(row, Mapping) else ""
        if not isinstance(row, Mapping):
            raise StageRefused("scene_stage_pose_missing", "unnamed")
        base = ROLE_PRIMS.get(role)
        if base is None:
            parts += 1
            prim = f"Part_{parts:02d}"
            role = role if _MODE_WORD.fullmatch(role) else "unnamed"
        else:
            seen[base] = seen.get(base, 0) + 1
            prim = base if seen[base] == 1 else f"{base}_{seen[base]:02d}"
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
        found.append({"role": role, "prim": prim, "name": package_name(prim), "usd_path": relative.as_posix(),
                      "position": [float(item) for item in position],
                      "orientation_xyzw": [float(item) for item in orientation],
                      "scale": [float(item) for item in scale] if scale is not None else None,
                      "visible": row.get("visible", True) is not False})
    return found


def package_name(prim: str) -> str:
    """The neutral file stem for a role prim: ``Object`` is ``object``, every other part goes under ``scene/``."""
    if prim == "Object":
        return "object"
    return "scene/" + re.sub(r"(?<=[a-z])(?=[A-Z])", "_", prim).lower()


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


def joint_prims(plan: Mapping[str, Any], packet: Path | None) -> dict[str, dict[str, str]]:
    """Each joint of the job's object file (the plan's ``task_object`` row), by its prim name: its path in scene.usda
    (``/World/Object/Asset/...``, where ``write_stage`` references the file) and its kind (``JOINT_KINDS``), both read
    from the file itself, never from the plan. A name two joints share is neither's. Empty without a readable file."""
    if packet is None:
        return {}
    try:
        row = next((row for row in placements(plan) if row["role"] == "task_object"), None)
    except StageRefused:
        return {}
    if row is None:
        return {}
    path = Path(packet).joinpath(*PurePosixPath(row["usd_path"]).parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(packet).resolve()):
        return {}
    try:
        from pxr import Sdf, Usd, UsdPhysics
        target = _target_prim(path, row["role"])
        stage = Usd.Stage.Open(str(path))
        root = Sdf.Path(target) if target else stage.GetDefaultPrim().GetPath()
        found: dict[str, dict[str, str]] = {}
        shared: set[str] = set()
        for prim in stage.Traverse():
            kind = ("prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else "revolute"
                    if prim.IsA(UsdPhysics.RevoluteJoint) else "fixed" if prim.IsA(UsdPhysics.FixedJoint) else None)
            if kind is None or not prim.GetPath().HasPrefix(root) or prim.GetPath() == root:
                continue
            composed = f"/World/{row['prim']}/Asset/{prim.GetPath().MakeRelativePath(root).pathString}"
            name = prim.GetName()
            if name in found or not _PRIM_PATH.fullmatch(composed):
                shared.add(name)
            found[name] = {"prim_path": composed, "kind": kind}
    except Exception:  # noqa: BLE001 - a file that cannot be read names no joint; the blocks are left out, counted
        return {}
    return {name: joint for name, joint in found.items() if name not in shared}


def _joint_blocks(blocks: Mapping[str, Any], joints: Mapping[str, Mapping[str, str]] | None) -> tuple[dict, int]:
    """The joint blocks (``JOINT_BLOCKS``) with each joint id as its prim path and kind (``joint_prims``), and how many
    fields were left out. Without ``joints`` a block is left out whole, as one field; a target joint that is not
    prismatic or revolute, or a locked joint the file does not name, is left out and counted."""
    out: dict[str, Any] = {}
    left_out = 0
    for key, block in blocks.items():
        if not joints or not isinstance(block, Mapping):
            left_out += 1
            continue
        if key == "target_joint":
            joint = joints.get(str(block.get("joint_id")))
            kind = (joint or {}).get("kind")
            if joint is None or kind not in ("prismatic", "revolute"):
                left_out += 1
                continue
            out[key] = {"prim_path": joint["prim_path"], "kind": kind, "unit": JOINT_KINDS[kind][0]}
            others = [item for item in block.get("joint_ids") or [] if item != block.get("joint_id")]
            left_out += len(others) + len(set(block) - {"joint_id", "joint_ids"})
            continue
        ids = block.get("joint_ids") if isinstance(block.get("joint_ids"), list) else []
        named = [joints[str(item)] for item in ids if str(item) in joints]
        kept: dict[str, Any] = {"joints": [{"prim_path": joint["prim_path"], "kind": joint["kind"]} for joint in named]}
        left_out += len(ids) - len(named)
        if block.get("mode") in MODE_WORDS:
            kept["mode"] = block["mode"]
        if _number_like(block.get("motion_tolerance")) and not isinstance(block.get("motion_tolerance"), bool):
            kept["motion_tolerance"] = block["motion_tolerance"]
        left_out += len(set(block) - {"joint_ids"} - set(kept))
        out[key] = kept
    return out, left_out


def rule_units(rule: Mapping[str, Any]) -> dict[str, str]:
    """Every unit the shipped rule uses, and what its windows count: each key suffix it carries
    (``SUFFIX_UNITS``, ``"*_m": "metres"``), ``*window_samples`` (control steps, at ``timing.control_frequency_hz``),
    and, for an articulated rule, the target joint's own numbers by their path (``TARGET_JOINT_FIELDS``: metres for a
    prismatic joint, radians for a revolute one) and the locked joints' tolerance."""
    units: dict[str, str] = {}
    windows = False

    def walk(value: Any) -> bool:
        """Whether ``value`` holds a number; each key that does names its unit."""
        nonlocal windows
        if isinstance(value, Mapping):
            numeric = False
            for key, item in value.items():
                holds = walk(item)
                numeric = numeric or holds
                if not holds or not isinstance(key, str):
                    continue
                windows = windows or key == "window_samples"
                suffix = next(((end, unit) for end, unit in SUFFIX_UNITS if key.endswith(end)), None)
                if suffix is not None:
                    units["*" + suffix[0]] = suffix[1]
            return numeric
        if isinstance(value, list):
            return any([walk(item) for item in value])
        return _number_like(value) and not isinstance(value, bool)

    walk({key: value for key, value in rule.items() if key not in ("units", "criteria_basis")})
    if windows:
        rate = (rule.get("timing") or {}).get("control_frequency_hz")
        units["*window_samples"] = (f"{WINDOW_SAMPLES}, {rate:g} a second (timing.control_frequency_hz)"
                                    if _number_like(rate) and not isinstance(rate, bool) else WINDOW_SAMPLES)
    criteria = rule.get("criteria") if isinstance(rule.get("criteria"), Mapping) else {}
    target = criteria.get("target_joint") if isinstance(criteria.get("target_joint"), Mapping) else {}
    unit, speed = JOINT_KINDS.get(str(target.get("kind")), (None, None))
    for fields, said, unknown in ((TARGET_JOINT_FIELDS, unit, UNKNOWN_JOINT_UNIT),
                                  (TARGET_JOINT_SPEED_FIELDS, speed, UNKNOWN_JOINT_SPEED)):
        for block, key in fields:
            if key in (criteria.get(block) or {}):
                units[f"criteria.{block}.{key}"] = f"{said} (the target joint is {target['kind']})" if said else unknown
    if "motion_tolerance" in (criteria.get("locked_joints") or {}):
        units["criteria.locked_joints.motion_tolerance"] = LOCKED_JOINT_UNIT
    return dict(sorted(units.items()))


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
        lines += [f'    def Xform "{row["prim"]}"', "    {"]
        if not row["visible"]:
            lines.append('        token visibility = "invisible"')
        lines += [f"        quatd xformOp:orient = ({_number(qw)}, {_number(qx)}, {_number(qy)}, {_number(qz)})"]
        if row.get("scale") is not None:
            sx, sy, sz = row["scale"]
            lines.append(f"        double3 xformOp:scale = ({_number(sx)}, {_number(sy)}, {_number(sz)})")
            order.append('"xformOp:scale"')
        lines += [f"        double3 xformOp:translate = ({_number(x)}, {_number(y)}, {_number(z)})",
                  f"        uniform token[] xformOpOrder = [{', '.join(order)}]", "",
                  '        def "Asset" (',
                  f"            prepend references = @./{relative.as_posix()}@{f'<{prim}>' if prim else ''}",
                  "        )", "        {", "        }", "    }"]
    lines += ["}", ""]
    text = "\n".join(lines)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(text, encoding="utf-8")
    return text


def _number_like(value: Any) -> bool:
    return isinstance(value, bool) or (isinstance(value, (int, float)) and math.isfinite(value))


def neutral_criteria(criteria: Mapping[str, Any], *, support_prim: str | None = None,
                     joints: Mapping[str, Mapping[str, str]] | None = None) -> tuple[dict[str, Any], int]:
    """The criteria through the allow-list, and how many fields were left out.

    ``support_prim_path`` names the support in Blueprint's scene; it becomes ``support_prim`` with the stage's own
    prim (``/World/Support``) when the stage has one, and is left out otherwise. ``joints``: the object file's joints
    (``joint_prims``); an articulated rule's joint blocks (``JOINT_BLOCKS``) ship each joint by its prim path and kind
    from them, and are left out without them.
    """
    blocks = {key: criteria[key] for key in JOINT_BLOCKS if isinstance(criteria, Mapping) and key in criteria}
    criteria = {key: value for key, value in criteria.items() if key not in blocks} if blocks else criteria
    named, left_out = _joint_blocks(blocks, joints)

    def keep(value: Any) -> Any:
        nonlocal left_out
        if _number_like(value):
            return value
        if isinstance(value, str):
            if value in MODE_WORDS:
                return value
            left_out += 1
            return None
        if isinstance(value, list):
            # A list of words keeps the known ones (each unknown word is left out and counted); a list that mixes
            # anything else must be whole or it goes.
            if value and all(isinstance(item, str) for item in value):
                kept_words = [item for item in value if item in MODE_WORDS]
                left_out += len(value) - len(kept_words)
                return kept_words if kept_words else None
            items = [keep(item) for item in value]
            if any(item is None for item in items):
                return None
            return items
        if isinstance(value, Mapping):
            out = {}
            for key, item in value.items():
                if key == "support_prim_path" and support_prim:
                    out["support_prim"] = support_prim
                    continue
                if key not in CRITERIA_KEYS:
                    left_out += 1
                    continue
                kept = keep(item)
                if kept is not None and kept != {}:
                    out[key] = kept
            return out
        left_out += 1
        return None
    kept = keep(criteria)
    return {**(kept if isinstance(kept, dict) else {}), **named}, left_out


def _range(value: Any) -> list[float] | None:
    if _finite(value, 2) and 0 <= value[0] <= value[1]:
        return [float(value[0]), float(value[1])]
    return None


def neutral_rule(plan: Mapping[str, Any], *, estimates: Mapping[str, Any] | None = None,
                 packet: Path | None = None, basis: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The success-rule file: the criteria and the neutral facts to reproduce the run, or None without criteria.

    ``estimates``: the preparation's ``physics.bounds`` for the object, estimated from the video; when it holds a
    mass or friction range they go under ``object_estimates_from_video``. ``packet``: as for ``up_axis``.
    ``basis``: for each number of the rule the preparation set, whether the site's own words state it
    (``site_stated``), it is the goal area's radius (``destination_area``) or Blueprint's default (``assumed``); it
    goes under ``criteria_basis``, by the preparation's field names. Never the site's words themselves.
    """
    criteria = ((plan.get("task_spec") or {}).get("task_success_contract") or {}).get("criteria")
    if not isinstance(criteria, Mapping):
        return None
    rows = placements(plan)
    support = "/World/Support" if any(row["prim"] == "Support" for row in rows) else None
    joints = joint_prims(plan, packet) if any(key in criteria for key in JOINT_BLOCKS) else None
    kept, left_out = neutral_criteria(criteria, support_prim=support, joints=joints)
    rule: dict[str, Any] = {"units": {}, "up_axis": up_axis(plan, packet=packet), "criteria": kept,
                            "criteria_fields_left_out": left_out}
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
            break
    extent = (plan.get("task_object_observability") or {}).get("task_object_extent_m")
    if _finite(extent, 3):
        rule["object_size_m"] = list(extent)
    if isinstance(estimates, Mapping):
        found = {key: _range(estimates.get(key)) for key in ("mass_kg", "static_friction", "dynamic_friction")}
        found = {key: value for key, value in found.items() if value is not None}
        if found:
            rule["object_estimates_from_video"] = found
    if isinstance(basis, Mapping):
        kept_basis = {str(key): value for key, value in basis.items()
                      if value in {"site_stated", "assumed", "destination_area"}}
        if kept_basis:
            rule["criteria_basis"] = kept_basis
    rule["units"] = rule_units(rule)
    return rule


__all__ = ["CRITERIA_KEYS", "JOINT_BLOCKS", "JOINT_KINDS", "MODE_WORDS", "ROLE_PRIMS", "SUFFIX_UNITS", "WORLD_FRAME_ROLES",
           "StageRefused", "joint_prims", "neutral_criteria", "neutral_rule", "package_name", "placements", "rule_units",
           "up_axis", "write_stage"]
