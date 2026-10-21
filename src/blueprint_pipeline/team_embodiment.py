"""A robot team's own embodiment: their robot model, the joints their policy drives, and their cameras.

Beta journey step: L5 team policy evaluation. A team brings the robot it would
bring to the site's pilot, whatever its form: a fixed arm, two arms, a mobile
manipulator, a wheeled or legged humanoid. Owner direction 2026-10-15: every
embodiment is evaluated, so nothing here is specific to one form. A robot is
described by two things the simulator needs and nothing else:

- **its base**: ``fixed`` (pinned to the world at the scene's robot station,
  the pose Blueprint placed its own arm at; the model authors the world anchor)
  or ``free`` (it stands or drives on the scene's floor under its own control;
  it starts at a clear floor stance facing the task, and falling ends the
  episode as the robot's failure);
- **its joints**: each driven joint, in action order, by ``position`` (arms,
  legs, torsos, heads, fingers) or by ``velocity`` (wheels), with its limits
  and optional gains; v1 optionally declares one gripper, while v2 declares
  ordered independent tools whose finger joints each share one 0-to-1 command.

plus one flattened USD (Isaac Sim's native format; Isaac Lab's URDF and MJCF
importers produce it) bound by SHA-256, the reset pose, the cameras the policy
sees (mounted on a link or in the world, OpenCV optical convention, resolution
and horizontal field of view), the control rate and the action chunk. A model
that lacks a declared joint or link, a fixed base that is not anchored or
cannot reach the task from the station, a free base that is anchored or has no
clear stance near the task, is refused by name before any GPU time is spent.

The task, the success contract and the scoring are unchanged: success is
decided from the task object's simulated state, sampled at the scene's task
rate, never from anything robot-side except the gripper's opening (from the
declared gripper joints and maximum width) and which declared links touch the
object.

The control plane validates, reads the model's bodies and joints (OpenUSD),
finds a free base's stance (``team_embodiment_stance``) and seals a binding
into the execution spec; the GPU worker applies the binding to each resolved
scene plan (``apply_team_embodiment``) before Isaac builds the scene, and
``team_embodiment_runtime`` drives it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "blueprint.robot_embodiment.v1"
SCHEMA_V2 = "blueprint.robot_embodiment.v2"
BINDING_SCHEMA = "blueprint.team_embodiment_binding.v1"
ROBOT_ID = "team_embodiment"
POLICY_INTERFACE_ID = "blueprint_policy_https.v1"
# What the team says its robot is (reported, never a gate); the base and joints decide how it is simulated.
FORMS = ("fixed_base_arm", "bimanual_arm", "mobile_manipulator", "wheeled_humanoid", "humanoid",
         "legged_manipulator", "other")
FIXED_FORMS = ("fixed_base_arm", "bimanual_arm")
BASE_MODES = ("fixed", "free")
JOINT_CONTROLS = ("position", "velocity")
MAX_JOINTS = 64
# Control rates: multiples of the scene's 15 Hz task-sampling rate that divide
# its 120 Hz physics rate, so every task sample lands on a control step. A
# slower policy repeats rows of its chunk.
CONTROL_HZ = (15, 30, 60, 120)
MAX_CAMERAS = 4
MAX_ACTION_CHUNK_ROWS = 64
BUNDLE_MODEL_PATH = "team_embodiment/robot.usd"
ENV_ROBOT = "{ENV_REGEX_NS}/Robot"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class TeamEmbodimentError(ValueError):
    """The team's embodiment cannot be evaluated; ``blockers`` names every reason."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = sorted({str(item) for item in blockers if str(item)})
        super().__init__(";".join(self.blockers))


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _vector(value: Any, length: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == length and all(_finite(item) for item in value)


def validate_embodiment_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """The manifest with its ``embodiment_digest``, or raise with every reason. No I/O."""

    m = dict(value) if isinstance(value, Mapping) else {}
    blockers: list[str] = []
    v2 = m.get("schema_version") == SCHEMA_V2
    allowed = {"schema_version", "embodiment_id", "form", "model", "base", "joints", "gripper", "tool_link",
               "reset_joint_positions", "actuator", "cameras", "control_hz", "action_chunk_rows", "reach_m",
               "embodiment_digest"}
    if v2:
        allowed = (allowed - {"gripper"}) | {"grippers", "arms"}
    if set(m) - allowed:
        blockers.append("team_embodiment_field_unknown:" + ",".join(sorted(set(m) - allowed)))
    if m.get("schema_version") not in (SCHEMA, SCHEMA_V2):
        blockers.append("team_embodiment_schema_invalid")
    if not _ID.fullmatch(str(m.get("embodiment_id") or "")):
        blockers.append("team_embodiment_id_invalid")
    if m.get("form") not in FORMS:
        blockers.append("team_embodiment_form_invalid")
    model = m.get("model") if isinstance(m.get("model"), Mapping) else {}
    url = urlsplit(str(model.get("url") or ""))
    if (model.get("format") != "usd" or url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or not _DIGEST.fullmatch(str(model.get("sha256") or ""))
            or set(model) != {"format", "url", "sha256"}):
        blockers.append("team_embodiment_model_invalid")
    base = m.get("base") if isinstance(m.get("base"), Mapping) else {}
    if base.get("mode") == "fixed":
        if set(base) != {"mode"}:
            blockers.append("team_embodiment_base_invalid")
    elif base.get("mode") == "free":
        root, fall, footprint = base.get("root_height_m"), base.get("fall_height_m"), base.get("footprint_half_extent_m")
        ground = base.get("ground_contact_links")
        if (set(base) != {"mode", "root_height_m", "fall_height_m", "footprint_half_extent_m", "ground_contact_links"}
                or not _finite(root) or not 0.05 <= float(root) <= 2.5
                or not _finite(fall) or not 0 < float(fall) < float(root)
                or not _vector(footprint, 2) or not all(0.05 <= float(v) <= 1.5 for v in footprint)
                or not isinstance(ground, list) or not 1 <= len(ground) <= 16
                or len(set(map(str, ground))) != len(ground) or not all(_NAME.fullmatch(str(v)) for v in ground)):
            blockers.append("team_embodiment_base_invalid")
    else:
        blockers.append("team_embodiment_base_invalid")
    if m.get("form") in FIXED_FORMS and base.get("mode") != "fixed":
        blockers.append("team_embodiment_form_base_mismatch")
    declared = m.get("joints") if isinstance(m.get("joints"), list) else []
    names: list[str] = []
    position_limits: dict[str, tuple[float, float]] = {}
    for joint in declared:
        joint = joint if isinstance(joint, Mapping) else {}
        control = joint.get("control")
        required = {"name", "control", "max_velocity"} | ({"lower", "upper"} if control == "position" else set())
        optional = {"stiffness", "damping"} if control == "position" else {"damping"}
        if (control not in JOINT_CONTROLS or not required <= set(joint) or set(joint) - required - optional
                or not _NAME.fullmatch(str(joint.get("name")))
                or not all(_finite(joint.get(key)) for key in required - {"name", "control"})
                or not float(joint["max_velocity"]) > 0
                or not all(_finite(joint[key]) and float(joint[key]) > 0 for key in optional & set(joint))
                or (control == "position" and not float(joint["lower"]) < float(joint["upper"]))):
            blockers.append("team_embodiment_joint_invalid")
            continue
        names.append(str(joint["name"]))
        if control == "position":
            position_limits[str(joint["name"])] = (float(joint["lower"]), float(joint["upper"]))
    if not 1 <= len(declared) <= MAX_JOINTS:
        blockers.append("team_embodiment_joints_count_invalid")
    gripper_names: list[str] = []
    if v2:
        channels = m.get("grippers")
        if not isinstance(channels, list) or len(channels) > 2:
            blockers.append("team_embodiment_grippers_invalid")
            channels = []
        channel_names = [g.get("name") if isinstance(g, Mapping) else None for g in channels]
        if (any(not _NAME.fullmatch(str(name or "")) for name in channel_names)
                or len(set(map(str, channel_names))) != len(channel_names)):
            blockers.append("team_embodiment_gripper_names_invalid")
        tool_links = [g.get("tool_link") if isinstance(g, Mapping) else None for g in channels]
        if (len(set(map(str, tool_links))) != len(tool_links)
                or (channels and m.get("tool_link") not in tool_links)):
            blockers.append("team_embodiment_gripper_tool_links_invalid")
        if m.get("form") == "bimanual_arm" and len(channels) != 2:
            blockers.append("team_embodiment_bimanual_tool_channels_required")
    else:
        channels = [] if m.get("gripper") is None else [m["gripper"]]
    for gripper in channels:
        gripper = gripper if isinstance(gripper, Mapping) else {}
        joints = gripper.get("joints") if isinstance(gripper.get("joints"), list) else []
        for joint in joints:
            joint = joint if isinstance(joint, Mapping) else {}
            if (set(joint) != {"name", "open", "closed"} or not _NAME.fullmatch(str(joint.get("name")))
                    or not _finite(joint.get("open")) or not _finite(joint.get("closed"))
                    or float(joint["open"]) == float(joint["closed"])):
                blockers.append("team_embodiment_gripper_joint_invalid")
                continue
            gripper_names.append(str(joint["name"]))
        contacts = gripper.get("contact_links")
        expected = {"joints", "max_width_m", "contact_links"} | ({"name", "tool_link"} if v2 else set())
        if v2 and not _NAME.fullmatch(str(gripper.get("tool_link") or "")):
            blockers.append("team_embodiment_gripper_tool_link_invalid")
        if (set(gripper) != expected or not 1 <= len(joints) <= 8
                or not _finite(gripper.get("max_width_m")) or not 0 < float(gripper["max_width_m"]) <= 1.0
                or not isinstance(contacts, list) or not 1 <= len(contacts) <= 8
                or len(set(map(str, contacts))) != len(contacts)
                or not all(_NAME.fullmatch(str(link)) for link in contacts)):
            blockers.append("team_embodiment_gripper_invalid")
    if len(set(names + gripper_names)) != len(names) + len(gripper_names):
        blockers.append("team_embodiment_joint_names_not_unique")
    if not _NAME.fullmatch(str(m.get("tool_link") or "")):
        blockers.append("team_embodiment_tool_link_invalid")
    if "arms" in m and v2:
        arms = m["arms"]
        if not isinstance(arms, list) or not 1 <= len(arms) <= 2 or base.get("mode") != "fixed":
            blockers.append("team_embodiment_arm_groups_invalid")
            arms = []
        used_names, used_joints, used_grippers = [], [], []
        channel_map = {g.get("name"): g for g in channels if isinstance(g, Mapping) and isinstance(g.get("name"), str)}
        controls = {j.get("name"): j.get("control") for j in declared if isinstance(j, Mapping)}
        for arm in arms:
            if not isinstance(arm, Mapping):
                blockers.append("team_embodiment_arm_group_invalid")
                continue
            joint_names = arm.get("joint_names")
            offset = arm.get("tool_from_tcp")
            channel = channel_map.get(arm.get("gripper_name")) if isinstance(arm.get("gripper_name"), str) else None
            if (set(arm) != {"name", "joint_names", "base_link", "tool_link", "gripper_name", "tool_from_tcp"}
                    or any(not _NAME.fullmatch(str(arm.get(key) or ""))
                           for key in ("name", "base_link", "tool_link", "gripper_name"))
                    or arm.get("base_link") == arm.get("tool_link")
                    or not isinstance(joint_names, list) or not 1 <= len(joint_names) <= MAX_JOINTS
                    or any(not isinstance(name, str) or controls.get(name) != "position" for name in joint_names)
                    or not channel or arm.get("tool_link") != channel.get("tool_link")
                    or not isinstance(offset, Mapping) or set(offset) != {"position_m", "orientation_xyzw"}
                    or not _vector(offset.get("position_m"), 3) or not _vector(offset.get("orientation_xyzw"), 4)
                    or not math.isclose(sum(float(q)**2 for q in offset["orientation_xyzw"]), 1.0, abs_tol=1e-4)):
                blockers.append("team_embodiment_arm_group_invalid")
                continue
            used_names.append(arm["name"])
            used_joints.extend(joint_names)
            used_grippers.append(arm["gripper_name"])
        if (len(set(used_names)) != len(used_names) or len(set(used_joints)) != len(used_joints)
                or len(set(used_grippers)) != len(used_grippers)
                or set(used_joints) != set(names) or set(used_grippers) != set(channel_map)):
            blockers.append("team_embodiment_arm_groups_ambiguous_or_incomplete")
    # Every position-driven joint (and gripper joint) has a reset position; a wheel starts at rest.
    reset = m.get("reset_joint_positions") if isinstance(m.get("reset_joint_positions"), Mapping) else {}
    if (set(reset) != set(position_limits) | set(gripper_names) or not all(_finite(v) for v in reset.values())
            or any(not low <= float(reset[name]) <= high for name, (low, high) in position_limits.items())):
        blockers.append("team_embodiment_reset_invalid")
    actuator = m.get("actuator") if isinstance(m.get("actuator"), Mapping) else {}
    if (set(actuator) != {"stiffness", "damping"}
            or not all(_finite(actuator.get(key)) and float(actuator[key]) > 0 for key in ("stiffness", "damping"))):
        blockers.append("team_embodiment_actuator_invalid")
    cameras = m.get("cameras") if isinstance(m.get("cameras"), list) else []
    seen_cameras = []
    for camera in cameras:
        camera = camera if isinstance(camera, Mapping) else {}
        mount = camera.get("mount_link")
        width, height, fov = camera.get("width"), camera.get("height"), camera.get("horizontal_fov_deg")
        if (set(camera) != {"name", "mount_link", "position_m", "orientation_xyzw", "width", "height",
                            "horizontal_fov_deg"} or not _NAME.fullmatch(str(camera.get("name")))
                or not (mount is None or _NAME.fullmatch(str(mount))) or not _vector(camera.get("position_m"), 3)
                or not _vector(camera.get("orientation_xyzw"), 4)
                or not math.isclose(sum(float(q) ** 2 for q in camera["orientation_xyzw"]), 1.0, abs_tol=1e-4)
                or type(width) is not int or type(height) is not int or not 32 <= width <= 1280
                or not 32 <= height <= 1280 or not _finite(fov) or not 10 <= float(fov) <= 170):
            blockers.append("team_embodiment_camera_invalid")
            continue
        seen_cameras.append(str(camera["name"]))
    if not 1 <= len(cameras) <= MAX_CAMERAS or len(set(seen_cameras)) != len(seen_cameras):
        blockers.append("team_embodiment_cameras_invalid")
    if m.get("control_hz") not in CONTROL_HZ:
        blockers.append("team_embodiment_control_hz_invalid")
    rows = m.get("action_chunk_rows")
    if type(rows) is not int or not 1 <= rows <= MAX_ACTION_CHUNK_ROWS:
        blockers.append("team_embodiment_action_chunk_rows_invalid")
    # A fixed base must reach the task from the station; a free base goes to it.
    if (base.get("mode") == "fixed") != ("reach_m" in m) or (
            "reach_m" in m and (not _finite(m["reach_m"]) or not 0.1 <= float(m["reach_m"]) <= 3.0)):
        blockers.append("team_embodiment_reach_invalid")
    if blockers:
        raise TeamEmbodimentError(blockers)
    manifest = {key: deepcopy(m[key]) for key in sorted(m) if key != "embodiment_digest"}
    manifest["embodiment_digest"] = canonical_digest(manifest, digest_field="embodiment_digest")
    if m.get("embodiment_digest") not in (None, manifest["embodiment_digest"]):
        raise TeamEmbodimentError(["team_embodiment_digest_mismatch"])
    return manifest


def gripper_channels(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ordered internal tool channels; never insert v2 fields into a legacy manifest."""

    if manifest.get("schema_version") == SCHEMA_V2:
        return [deepcopy(dict(channel)) for channel in manifest["grippers"]]
    gripper = manifest.get("gripper")
    return [] if not gripper else [{"name": "gripper", "tool_link": manifest["tool_link"], **deepcopy(dict(gripper))}]


def action_dimension(manifest: Mapping[str, Any]) -> int:
    """Declared joint targets, followed by one closure per ordered tool (0 to 1)."""

    return len(manifest["joints"]) + len(gripper_channels(manifest))


def camera_names(manifest: Mapping[str, Any]) -> list[str]:
    return [str(camera["name"]) for camera in manifest["cameras"]]


def verify_model_file(manifest: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """The staged robot USD's identity, or raise when it is not the declared file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TeamEmbodimentError(["team_embodiment_model_missing"])
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        head = stream.read(64)
        digest.update(head)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if "sha256:" + digest.hexdigest() != manifest["model"]["sha256"]:
        raise TeamEmbodimentError(["team_embodiment_model_digest_mismatch"])
    if not (head.startswith(b"PXR-USDC") or head.lstrip().startswith(b"#usda")):
        raise TeamEmbodimentError(["team_embodiment_model_not_usd"])
    return {"sha256": manifest["model"]["sha256"], "size_bytes": source.stat().st_size}


def validate_model_dependency_closure(value: Mapping[str, Any], *, model_record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a portable, fully inventoried prepared directory; never resolve caller paths."""
    if (not isinstance(value, Mapping)
            or set(value) != {"schema_version", "model_path", "files", "closure_digest"}
            or value.get("schema_version") != "embodiment_asset_dependency_closure.v1"
            or value.get("closure_digest") != cross_runtime_canonical_digest(dict(value), digest_field="closure_digest")):
        raise TeamEmbodimentError(["team_embodiment_dependency_closure_invalid"])
    files = value.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 10_000:
        raise TeamEmbodimentError(["team_embodiment_dependency_closure_invalid"])
    paths, total = [], 0
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "size_bytes"}:
            raise TeamEmbodimentError(["team_embodiment_dependency_closure_invalid"])
        path, size = row.get("path"), row.get("size_bytes")
        pure = PurePosixPath(path) if isinstance(path, str) else PurePosixPath()
        if (not isinstance(path, str) or not path or not pure.parts or path != pure.as_posix() or pure.is_absolute()
                or any(part in {"..", "."} for part in pure.parts) or "\\" in path or "\x00" in path
                or not _DIGEST.fullmatch(str(row.get("sha256") or ""))
                or type(size) is not int or not 0 <= size <= 512 * 1024 * 1024):
            raise TeamEmbodimentError(["team_embodiment_dependency_closure_invalid"])
        paths.append(path)
        total += size
    if paths != sorted(set(paths)) or total > 2 * 1024**3:
        raise TeamEmbodimentError(["team_embodiment_dependency_closure_invalid"])
    root = next((row for row in files if row["path"] == value["model_path"]), None)
    if root is None or any(root[key] != model_record.get(key) for key in ("sha256", "size_bytes")):
        raise TeamEmbodimentError(["team_embodiment_dependency_closure_model_mismatch"])
    return deepcopy(dict(value))


def validate_source_import_provenance(value: Any, *, model_record: Mapping[str, Any],
                                      dependency_closure: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the complete verified importer receipt bound to its exact model.

    Staging verifies the prepared directory itself. This transport validator
    neither rereads a source path nor promotes imported assets to qualification.
    Its schema value matches deployment_embodiment_asset_import.SCHEMA without
    importing the control-plane importer into the provider runtime closure.
    """
    error = "team_embodiment_source_import_provenance_invalid"
    try:
        if not isinstance(value, Mapping):
            raise ValueError(error)
        raw = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError(error)
        receipt = json.loads(raw)
        model = receipt.get("model")
        if (receipt.get("schema_version") != "embodiment_asset_import.v1"
                or receipt.get("claim_ceiling") != "source_asset_not_qualified"
                or receipt.get("execution_authorized") is not False
                or receipt.get("prepared_directory_self_contained") is not True
                or receipt.get("source_bundle_required_for_dependency_resolution") is not False
                or type(receipt.get("fix_base")) is not bool
                or not _DIGEST.fullmatch(str(receipt.get("source_bundle_digest") or ""))
                or receipt.get("import_digest") != cross_runtime_canonical_digest(receipt, digest_field="import_digest")
                or dependency_closure is None or receipt.get("dependency_closure") != dependency_closure
                or not isinstance(model, Mapping) or set(model) != {"path", "sha256", "size_bytes"}
                or model["path"] != dependency_closure["model_path"]
                or type(model["size_bytes"]) is not int
                or any(model[key] != model_record.get(key) for key in ("sha256", "size_bytes"))
                or not isinstance(receipt.get("files"), list) or model not in receipt["files"]):
            raise ValueError(error)
        validate_model_dependency_closure(receipt["dependency_closure"], model_record=model_record)
    except (TypeError, ValueError, KeyError) as exc:
        raise TeamEmbodimentError([error]) from exc
    return receipt


def read_model_structure(path: str | Path) -> dict[str, Any]:
    """The robot USD's rigid-body paths (as spawned under the robot prim) and joint names (OpenUSD)."""

    from .native_task_robot_contact_topology import (
        NativeTaskRobotContactTopologyError,
        _asset_rigid_body_paths,
    )

    try:
        bodies = _asset_rigid_body_paths(Path(path))
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(path))
        joints = sorted({prim.GetName() for prim in stage.Traverse()
                         if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)})
        # A fixed joint with one empty body pins that link to the world. Isaac Lab's
        # fix_root_link cannot do it for this topology (root Xform, links beneath),
        # so the model must author its own anchor, as Isaac's URDF importer does.
        anchors = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(prim)
            ends = [list(joint.GetBody0Rel().GetTargets()), list(joint.GetBody1Rel().GetTargets())]
            linked = [stage.GetPrimAtPath(targets[0]) for targets in ends if targets]
            if len(linked) == 1 and linked[0] and linked[0].HasAPI(UsdPhysics.RigidBodyAPI):
                anchors.append(linked[0].GetName())
    except (NativeTaskRobotContactTopologyError, Exception) as exc:  # pxr raises Tf errors on bad bytes
        raise TeamEmbodimentError(["team_embodiment_model_unreadable"]) from exc
    return {"body_paths": bodies, "joint_names": joints, "world_anchor_links": sorted(anchors)}


def _body_path(structure: Mapping[str, Any], link: str) -> str | None:
    matches = [path for path in structure["body_paths"] if path.rsplit("/", 1)[-1] == link]
    return matches[0] if len(matches) == 1 else None


def model_structure_blockers(manifest: Mapping[str, Any], structure: Mapping[str, Any]) -> list[str]:
    """Every declared joint and link exists, once, in the model; a fixed base is anchored, a free one is not."""

    channels = gripper_channels(manifest)
    joints = [j["name"] for j in manifest["joints"]] + [j["name"] for g in channels for j in g["joints"]]
    links = [manifest["tool_link"], *(g["tool_link"] for g in channels),
             *(link for g in channels for link in g["contact_links"]),
             *(link for arm in manifest.get("arms", []) for link in (arm["base_link"], arm["tool_link"])),
             *(c["mount_link"] for c in manifest["cameras"] if c["mount_link"] is not None),
             *manifest["base"].get("ground_contact_links", [])]
    blockers = [f"team_embodiment_model_joint_missing:{name}" for name in joints
                if name not in structure["joint_names"]]
    blockers += [f"team_embodiment_model_link_missing_or_ambiguous:{link}" for link in dict.fromkeys(links)
                 if _body_path(structure, link) is None]
    # A fixed base whose model does not pin it floats (contacts would move it mid-episode);
    # a free base whose model pins it could never stand, walk or drive.
    anchored = bool(structure.get("world_anchor_links"))
    if manifest["base"]["mode"] == "fixed" and not anchored:
        blockers.append("team_embodiment_model_base_not_anchored_to_world")
    if manifest["base"]["mode"] == "free" and anchored:
        blockers.append("team_embodiment_model_free_base_anchored_to_world")
    return blockers


def station_reach_blockers(scene_plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """Refuse a fixed base whose declared reach cannot cover the task from the scene's robot station."""

    if manifest["base"]["mode"] != "fixed":
        return []
    robot = scene_plan.get("robot") if isinstance(scene_plan.get("robot"), Mapping) else {}
    base = ((robot.get("base_pose_world") or {}).get("position_world_m")
            if isinstance(robot.get("base_pose_world"), Mapping) else None)
    task_spec = scene_plan.get("task_spec") if isinstance(scene_plan.get("task_spec"), Mapping) else {}
    if not _vector(base, 3):
        return ["team_embodiment_station_unavailable"]
    points = []
    start = task_spec.get("start_pose_world")
    if isinstance(start, list) and len(start) >= 3 and _vector(start[:3], 3):
        points.append([float(v) for v in start[:3]])
    box = task_spec.get("destination_position_bounds_world_m")
    if isinstance(box, Mapping) and _vector(box.get("minimum"), 3) and _vector(box.get("maximum"), 3):
        points.append([(float(box["minimum"][i]) + float(box["maximum"][i])) / 2 for i in range(3)])
    if not points:
        return ["team_embodiment_station_task_points_unavailable"]
    reach = float(manifest["reach_m"])
    return ([] if all(math.dist([float(v) for v in base], point) <= reach for point in points)
            else ["team_embodiment_station_out_of_reach"])


def team_embodiment_binding(manifest: Mapping[str, Any], *, model_record: Mapping[str, Any],
                            structure: Mapping[str, Any], scene_plan: Mapping[str, Any],
                            stance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The sealed binding the execution spec carries, or raise with every reason (no I/O).

    A free base carries its floor ``stance`` (``team_embodiment_stance.free_base_stance``).
    """

    free = manifest["base"]["mode"] == "free"
    blockers = [*model_structure_blockers(manifest, structure), *station_reach_blockers(scene_plan, manifest)]
    if free and not (isinstance(stance, Mapping) and _vector(stance.get("position_world_m"), 3)
                     and _vector(stance.get("orientation_xyzw"), 4) and _finite(stance.get("floor_z_m"))):
        blockers.append("team_embodiment_free_base_stance_unavailable")
    if blockers:
        raise TeamEmbodimentError(blockers)
    closure = (validate_model_dependency_closure(model_record["dependency_closure"], model_record=model_record)
               if "dependency_closure" in model_record else None)
    provenance = (validate_source_import_provenance(model_record["source_import_provenance"], model_record=model_record,
                                                  dependency_closure=closure)
                  if "source_import_provenance" in model_record else None)
    binding = {"schema_version": BINDING_SCHEMA, "manifest": deepcopy(dict(manifest)),
               "model": {"sha256": model_record["sha256"], "size_bytes": int(model_record["size_bytes"]),
                         "bundle_path": "team_embodiment/" + closure["model_path"] if closure else BUNDLE_MODEL_PATH,
                         **({"dependency_closure": closure} if closure else {}),
                         **({"source_import_provenance": provenance} if provenance else {})},
               "body_paths": list(structure["body_paths"]),
               **({"stance": deepcopy(dict(stance))} if free else {}), "binding_digest": ""}
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    return binding


def camera_slots(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(role, pose_frame)`` for each declared camera, in manifest order; the role is its scene name."""

    return [(f"team_camera_{index}", "world" if camera["mount_link"] is None else "robot_body")
            for index, camera in enumerate(manifest["cameras"])]


def _matrix(position: Sequence[float], xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = (float(v) for v in xyzw)
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [*rotation[0], float(position[0]), *rotation[1], float(position[1]), *rotation[2], float(position[2]),
            0.0, 0.0, 0.0, 1.0]


def _plan_cameras(manifest: Mapping[str, Any], structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for (role, frame), camera in zip(camera_slots(manifest), manifest["cameras"], strict=True):
        width, height = int(camera["width"]), int(camera["height"])
        focal = (width / 2.0) / math.tan(math.radians(float(camera["horizontal_fov_deg"])) / 2.0)
        rows.append({
            "role": role, "team_camera_name": camera["name"], "pose_frame": frame, "optical_convention": "opencv",
            "parent_prim_path": ("{ENV_REGEX_NS}" if frame == "world"
                                 else _body_path(structure, camera["mount_link"])),
            "frame_from_camera_matrix": _matrix(camera["position_m"], camera["orientation_xyzw"]),
            "intrinsics": {"fx": focal, "fy": focal, "cx": (width - 1) / 2.0, "cy": (height - 1) / 2.0,
                           "width": width, "height": height},
            "policy_input": True, "review_only": False,
        })
    return rows


def _shift_team_cameras(plan: dict[str, Any]) -> None:
    """A camera-shifted cell moves the team's fixed cameras, not the Blueprint camera they replaced.

    The scene resolver shifted Blueprint's ``external`` camera, which the team's
    cameras replace. The same shift goes to every team camera fixed in the world
    (each mounted camera when the team declared none), and the readback checks
    each one.
    """

    scenario = plan.get("scenario") if isinstance(plan.get("scenario"), dict) else {}
    applications = list(scenario.get("parameter_applications") or [])
    shifts = [row for row in applications if row.get("parameter_id") == "external_camera_x_delta_m"]
    if not shifts:
        return
    delta = float(shifts[0]["delta_from_nominal"])
    team = [row for row in plan["cameras"] if row.get("policy_input") is True]
    moved = [row for row in team if row["pose_frame"] == "world"] or team
    kept = [row for row in applications if row.get("parameter_id") != "external_camera_x_delta_m"]
    for camera in moved:
        camera["frame_from_camera_matrix"][3] += delta
        kept.append({**shifts[0], "camera_role": camera["role"],
                     "expected_native_value": camera["frame_from_camera_matrix"][3]})
    scenario["parameter_applications"] = kept


def _rebind_contact_sensors(articulation: dict[str, Any], *, task_bodies: list[str], protected: list[str]) -> None:
    """Point the scene compiler's robot-side contact sensors at the team robot's bodies."""

    forbidden = sorted(set(protected) - set(task_bodies))
    sensors = list(articulation.get("contact_sensors") or [])
    scene_filter = next((sensor["filter_prim_paths_expr"] for sensor in sensors
                         if sensor.get("logical_sensor_id") == "robot_scene_contact"), None)
    kept = []
    for sensor in sensors:
        logical = sensor.get("logical_sensor_id")
        if logical == "robot_scene_contact":
            continue
        if logical == "task_robot_contact":
            sensor = {**sensor, "filter_prim_paths_expr": task_bodies}
        elif logical == "robot_task_forbidden_collision":
            if not forbidden:
                continue
            sensor = {**sensor, "filter_prim_paths_expr": forbidden}
        kept.append(sensor)
    if scene_filter is not None:
        kept.extend({"sensor_instance_id": f"robot_scene_contact__{index:02d}",
                     "logical_sensor_id": "robot_scene_contact", "prim_path": body,
                     "filter_prim_paths_expr": scene_filter} for index, body in enumerate(protected))
    articulation["contact_sensors"] = kept
    if isinstance(articulation.get("robot_contact_topology"), Mapping):
        articulation["robot_contact_topology"] = {
            **articulation["robot_contact_topology"], "robot_id": ROBOT_ID,
            "task_contact_body_paths": task_bodies, "protected_collision_body_paths": protected}
    if "forbidden_robot_contact_body_paths" in articulation:
        articulation["forbidden_robot_contact_body_paths"] = forbidden


def apply_team_embodiment(plan: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """Put the team's robot at the scene's robot station: robot, cameras, cadence and contact sensors."""

    if (binding.get("schema_version") != BINDING_SCHEMA
            or binding.get("binding_digest") != canonical_digest(dict(binding), digest_field="binding_digest")):
        raise TeamEmbodimentError(["team_embodiment_binding_invalid"])
    manifest = validate_embodiment_manifest(binding["manifest"])
    closure = None
    if "dependency_closure" in binding["model"]:
        closure = validate_model_dependency_closure(binding["model"]["dependency_closure"], model_record=binding["model"])
        if binding["model"]["bundle_path"] != "team_embodiment/" + closure["model_path"]:
            raise TeamEmbodimentError(["team_embodiment_dependency_closure_model_mismatch"])
    provenance = (validate_source_import_provenance(binding["model"]["source_import_provenance"],
                                                  model_record=binding["model"], dependency_closure=closure)
                  if "source_import_provenance" in binding["model"] else None)
    structure = {"body_paths": list(binding["body_paths"])}
    value = deepcopy(dict(plan))
    station = (value.get("robot") or {}).get("base_pose_world")
    if not isinstance(station, Mapping):
        raise TeamEmbodimentError(["team_embodiment_station_unavailable"])
    channels = gripper_channels(manifest)
    gripper = manifest.get("gripper") or {}
    contact_links = list(dict.fromkeys(link for g in channels for link in g["contact_links"])) or [manifest["tool_link"]]
    task_bodies = sorted({_body_path(structure, link) for link in contact_links} - {None})
    if len(task_bodies) != len(contact_links):
        raise TeamEmbodimentError(["team_embodiment_contact_links_not_in_model"])
    reset = {name: float(position) for name, position in manifest["reset_joint_positions"].items()}
    joints, defaults = manifest["joints"], manifest["actuator"]
    base = manifest["base"]
    if base["mode"] == "free":
        stance = binding["stance"]
        base_row = {"base_mode": "free_on_floor",
                    "base_pose_world": {"position_world_m": list(stance["position_world_m"]),
                                        "orientation_xyzw": list(stance["orientation_xyzw"])},
                    "fall_height_world_m": float(stance["floor_z_m"]) + float(base["fall_height_m"])}
    else:
        base_row = {"base_mode": "fixed_at_station", "base_pose_world": deepcopy(dict(station))}
    value["robot"] = {
        "robot_id": ROBOT_ID, "embodiment_id": manifest["embodiment_id"],
        "embodiment_digest": manifest["embodiment_digest"], "form": manifest["form"], **base_row,
        "usd_path": binding["model"]["bundle_path"], "usd_sha256": binding["model"]["sha256"],
        "usd_size_bytes": int(binding["model"]["size_bytes"]), "usd_dependency_bindings": [],
        "usd_runtime_asset_dependencies": [],
        "joint_names": [joint["name"] for joint in joints],
        "joint_controls": {joint["name"]: joint["control"] for joint in joints},
        "joint_position_limits_rad": {joint["name"]: [float(joint["lower"]), float(joint["upper"])]
                                      for joint in joints if joint["control"] == "position"},
        "joint_velocity_limits": {joint["name"]: float(joint["max_velocity"]) for joint in joints},
        # A wheel is driven by velocity: no stiffness, only damping.
        "joint_gains": {joint["name"]: {
            "stiffness": 0.0 if joint["control"] == "velocity" else float(joint.get("stiffness", defaults["stiffness"])),
            "damping": float(joint.get("damping", defaults["damping"]))} for joint in joints},
        "gripper_joints": [dict(joint) for joint in gripper.get("joints") or []],
        "gripper_max_width_m": gripper.get("max_width_m"),
        "joint_reset_positions_rad": reset, "actuator": dict(manifest["actuator"]),
        "tool_link": manifest["tool_link"], "camera_names": camera_names(manifest),
        "task_contact_body_paths": task_bodies,
        # The rigid readback's grasp position is the midpoint of two bodies; one contact link counts twice.
        "grasp_frame": {"kind": "body_midpoint", "body_names": (contact_links * 2)[:2]},
        "control_hz": int(manifest["control_hz"]), "action_chunk_rows": int(manifest["action_chunk_rows"]),
    }
    if closure is not None:
        value["robot"]["usd_dependency_closure"] = deepcopy(closure)
        value["robot"]["usd_dependency_bindings"] = [
            {"relative_path": "team_embodiment/" + record["path"],
             "sha256": record["sha256"], "size_bytes": record["size_bytes"]}
            for record in closure["files"] if record["path"] != closure["model_path"]]
    if provenance is not None:
        value["robot"]["source_import_provenance"] = deepcopy(provenance)
    if manifest["schema_version"] == SCHEMA_V2:
        # Keep each channel's measured geometry and command order. The legacy
        # flattened joints are solely the simulator's position-action order.
        value["robot"].update(
            enabled_self_collisions=True,
            gripper_joints=[dict(joint) for channel in channels for joint in channel["joints"]],
            gripper_channels=[{**deepcopy(channel),
                               "task_contact_body_paths": [_body_path(structure, link)
                                                           for link in channel["contact_links"]]}
                              for channel in channels],
        )
        # The primary legacy readback remains primary: never average tools.
        primary = next((g for g in channels if g["tool_link"] == manifest["tool_link"]), None)
        primary_links = primary["contact_links"] if primary else [manifest["tool_link"]]
        value["robot"]["grasp_frame"]["body_names"] = (primary_links * 2)[:2]
        if "arms" in manifest:
            value["robot"]["arms"] = deepcopy(manifest["arms"])
    if isinstance(value.get("reset"), dict):
        value["reset"]["robot_joint_positions_rad"] = dict(reset)
    # The team's cameras replace Blueprint's policy cameras; the review (overview) camera stays.
    review = [row for row in value.get("cameras") or [] if row.get("review_only") is True]
    value["cameras"] = [*_plan_cameras(manifest, structure), *review]
    _shift_team_cameras(value)
    cadence = value["cadence"]
    task_hz = float(value["task_spec"]["control_frequency_hz"])
    hz = int(manifest["control_hz"])
    physics_hz = float(cadence["physics_frequency_hz"])
    if hz % task_hz or physics_hz % hz:
        raise TeamEmbodimentError(["team_embodiment_control_hz_incompatible_with_scene"])
    stride = int(hz // task_hz)
    cadence.update(control_frequency_hz=float(hz), control_decimation=int(physics_hz // hz),
                   task_sample_stride=stride,
                   maximum_action_steps=int(cadence["maximum_action_steps"]) * stride,
                   settle_window_samples=int(cadence["settle_window_samples"]) * stride)
    if isinstance(value.get("articulation"), dict):
        # Feet and wheels bear the robot's weight on the floor: that is standing, not a collision.
        ground = {_body_path(structure, link) for link in base.get("ground_contact_links", [])}
        _rebind_contact_sensors(value["articulation"], task_bodies=task_bodies,
                                protected=sorted(set(binding["body_paths"]) - ground))
        if manifest["schema_version"] == SCHEMA_V2:
            sensors = value["articulation"]["contact_sensors"]
            # Separate real filtered sensors, sharing the existing logical
            # aggregate. Each tool retains exactly which instances measured it.
            expanded = []
            for sensor in sensors:
                if sensor["logical_sensor_id"] != "task_robot_contact":
                    expanded.append(sensor)
                    continue
                for channel in value["robot"]["gripper_channels"]:
                    sensor_id = sensor["sensor_instance_id"] + "__tool_" + channel["name"]
                    expanded.append({**sensor, "sensor_instance_id": sensor_id,
                                     "filter_prim_paths_expr": channel["task_contact_body_paths"]})
                    channel.setdefault("task_contact_sensor_instances", []).append(sensor_id)
            value["articulation"]["contact_sensors"] = expanded
    profile = {"schema_version": "team_embodiment_profile.v1", "robot_preset_id": ROBOT_ID,
               "embodiment_id": manifest["embodiment_id"], "embodiment_digest": manifest["embodiment_digest"],
               "policy_interface_id": POLICY_INTERFACE_ID,
               "policy_camera_roles": [role for role, _ in camera_slots(manifest)],
               "review_only_camera_roles": [row.get("role") for row in review],
               "claim_boundary": {"base": ("held_fixed_at_scene_robot_station" if base["mode"] == "fixed"
                                           else "free_on_scene_floor_from_clear_stance_falling_fails_episode"),
                                  "task_sampling_hz": task_hz, "control_hz": hz},
               "binding_digest": binding["binding_digest"], "profile_digest": ""}
    if manifest["schema_version"] == SCHEMA_V2:
        profile["claim_boundary"]["self_collision_mode"] = {
            "enabled": True, "force_readback": "task_and_scene_only_no_arm_arm_force_channel",
            "qualified": False}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    value["policy_canary_embodiment_profile"] = profile
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def is_team_plan(plan: Mapping[str, Any]) -> bool:
    return (plan.get("robot") or {}).get("robot_id") == ROBOT_ID


__all__ = [
    "BASE_MODES",
    "BINDING_SCHEMA",
    "BUNDLE_MODEL_PATH",
    "CONTROL_HZ",
    "FIXED_FORMS",
    "FORMS",
    "POLICY_INTERFACE_ID",
    "ROBOT_ID",
    "SCHEMA",
    "SCHEMA_V2",
    "gripper_channels",
    "TeamEmbodimentError",
    "action_dimension",
    "apply_team_embodiment",
    "camera_names",
    "camera_slots",
    "is_team_plan",
    "model_structure_blockers",
    "read_model_structure",
    "station_reach_blockers",
    "team_embodiment_binding",
    "validate_embodiment_manifest",
    "validate_model_dependency_closure",
    "validate_source_import_provenance",
    "verify_model_file",
]
