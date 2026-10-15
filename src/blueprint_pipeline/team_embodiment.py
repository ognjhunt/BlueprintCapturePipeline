"""A robot team's own embodiment: their robot model, the joints their policy drives, and their cameras.

Beta journey step: L5 team policy evaluation. Until now the only robot a team
could be evaluated on was Blueprint's simulated Franka Panda with a Robotiq
2F-85 on the DROID interface. A team that deploys a UR5e, a Kinova, an xArm or
any other arm would have been evaluated on a robot it would never bring to the
site's pilot. This module lets the team bring its own:

- **the robot**: one flattened USD (Isaac Sim's native format; Isaac Lab's
  URDF importer produces it from a URDF), bound by SHA-256, plus the joints the
  policy drives in action order with their limits, the gripper joints and their
  open and closed positions, the links that may touch the task object, the tool
  link, the reset pose and the actuator gains;
- **the cameras** the policy sees: each mounted on a robot link or in the world,
  posed in the OpenCV optical convention (+Z forward, +X right, +Y down) with
  its resolution and horizontal field of view;
- **the control rate** and the action chunk its policy returns.

The robot's base is held fixed at the scene's robot station, the pose Blueprint
placed its own robot at when it built the site's scene. A robot whose declared
reach does not cover the task object and its destination from the station, or
whose model lacks a declared joint or link, is refused by name before any GPU
time is spent. Fixed-base arms are the admitted form: evaluating a mobile
manipulator or a humanoid with its base locked needs a recorded owner focus
decision first (``ADMITTED_FORMS``).

The task, the success contract and the scoring are unchanged: success is
decided from the task object's simulated state, sampled at the scene's task
rate, never from anything robot-side except the gripper's opening (from the
declared gripper joints and maximum width) and which declared links touch the
object.

The control plane validates, reads the model's bodies and joints (OpenUSD) and
seals a binding into the execution spec; the GPU worker applies the binding to
each resolved scene plan (``apply_team_embodiment``) before Isaac builds the
scene, and ``team_embodiment_runtime`` drives it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest

SCHEMA = "blueprint.robot_embodiment.v1"
BINDING_SCHEMA = "blueprint.team_embodiment_binding.v1"
ROBOT_ID = "team_embodiment"
POLICY_INTERFACE_ID = "blueprint_policy_https.v1"
FORMS = ("fixed_base_arm", "mobile_manipulator", "humanoid")
# Forms the runtime will evaluate today. Others are declared truthfully and refused by name.
ADMITTED_FORMS = ("fixed_base_arm",)
# Control rates: multiples of the scene's 15 Hz task-sampling rate that divide
# its 120 Hz physics rate, so every task sample lands on a control step. A
# slower policy repeats rows of its chunk.
CONTROL_HZ = (15, 30, 60)
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
    allowed = {"schema_version", "embodiment_id", "form", "model", "arm_joints", "gripper", "tool_link",
               "reset_joint_positions", "actuator", "cameras", "control_hz", "action_chunk_rows", "reach_m",
               "embodiment_digest"}
    if set(m) - allowed:
        blockers.append("team_embodiment_field_unknown:" + ",".join(sorted(set(m) - allowed)))
    if m.get("schema_version") != SCHEMA:
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
    arm = m.get("arm_joints") if isinstance(m.get("arm_joints"), list) else []
    names: list[str] = []
    for joint in arm:
        joint = joint if isinstance(joint, Mapping) else {}
        if (set(joint) != {"name", "lower", "upper", "max_velocity"} or not _NAME.fullmatch(str(joint.get("name")))
                or not all(_finite(joint.get(key)) for key in ("lower", "upper", "max_velocity"))
                or not float(joint["lower"]) < float(joint["upper"]) or not float(joint["max_velocity"]) > 0):
            blockers.append("team_embodiment_arm_joint_invalid")
            continue
        names.append(str(joint["name"]))
    if not 1 <= len(arm) <= 32:
        blockers.append("team_embodiment_arm_joints_count_invalid")
    gripper = m.get("gripper")
    gripper_names: list[str] = []
    if gripper is not None:
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
        if (set(gripper) != {"joints", "max_width_m", "contact_links"} or not 1 <= len(joints) <= 8
                or not _finite(gripper.get("max_width_m")) or not 0 < float(gripper["max_width_m"]) <= 1.0
                or not isinstance(contacts, list) or not 1 <= len(contacts) <= 8
                or len(set(map(str, contacts))) != len(contacts)
                or not all(_NAME.fullmatch(str(link)) for link in contacts)):
            blockers.append("team_embodiment_gripper_invalid")
    if len(set(names + gripper_names)) != len(names) + len(gripper_names):
        blockers.append("team_embodiment_joint_names_not_unique")
    if not _NAME.fullmatch(str(m.get("tool_link") or "")):
        blockers.append("team_embodiment_tool_link_invalid")
    reset = m.get("reset_joint_positions") if isinstance(m.get("reset_joint_positions"), Mapping) else {}
    limits = {str(j["name"]): (float(j["lower"]), float(j["upper"])) for j in arm
              if isinstance(j, Mapping) and str(j.get("name")) in names}
    if (set(reset) != set(names) | set(gripper_names) or not all(_finite(v) for v in reset.values())
            or any(not limits[name][0] <= float(reset[name]) <= limits[name][1] for name in names)):
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
    if not _finite(m.get("reach_m")) or not 0.1 <= float(m["reach_m"]) <= 3.0:
        blockers.append("team_embodiment_reach_invalid")
    if blockers:
        raise TeamEmbodimentError(blockers)
    manifest = {key: deepcopy(m[key]) for key in sorted(m) if key != "embodiment_digest"}
    manifest["embodiment_digest"] = canonical_digest(manifest, digest_field="embodiment_digest")
    if m.get("embodiment_digest") not in (None, manifest["embodiment_digest"]):
        raise TeamEmbodimentError(["team_embodiment_digest_mismatch"])
    return manifest


def action_dimension(manifest: Mapping[str, Any]) -> int:
    """One target per driven arm joint, then the gripper's closure fraction (0 open, 1 closed)."""

    return len(manifest["arm_joints"]) + (1 if manifest.get("gripper") else 0)


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
    except (NativeTaskRobotContactTopologyError, Exception) as exc:  # pxr raises Tf errors on bad bytes
        raise TeamEmbodimentError(["team_embodiment_model_unreadable"]) from exc
    return {"body_paths": bodies, "joint_names": joints}


def _body_path(structure: Mapping[str, Any], link: str) -> str | None:
    matches = [path for path in structure["body_paths"] if path.rsplit("/", 1)[-1] == link]
    return matches[0] if len(matches) == 1 else None


def model_structure_blockers(manifest: Mapping[str, Any], structure: Mapping[str, Any]) -> list[str]:
    """Every declared joint and link must exist, once, in the model."""

    gripper = manifest.get("gripper") or {}
    joints = [j["name"] for j in manifest["arm_joints"]] + [j["name"] for j in gripper.get("joints") or []]
    links = [manifest["tool_link"], *(gripper.get("contact_links") or []),
             *(c["mount_link"] for c in manifest["cameras"] if c["mount_link"] is not None)]
    blockers = [f"team_embodiment_model_joint_missing:{name}" for name in joints
                if name not in structure["joint_names"]]
    blockers += [f"team_embodiment_model_link_missing_or_ambiguous:{link}" for link in dict.fromkeys(links)
                 if _body_path(structure, link) is None]
    return blockers


def station_reach_blockers(scene_plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """Refuse a robot whose declared reach cannot cover the task from the scene's robot station."""

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


def admission_blockers(manifest: Mapping[str, Any]) -> list[str]:
    return [] if manifest["form"] in ADMITTED_FORMS else [f"team_embodiment_form_not_admitted:{manifest['form']}"]


def team_embodiment_binding(manifest: Mapping[str, Any], *, model_record: Mapping[str, Any],
                            structure: Mapping[str, Any], scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """The sealed binding the execution spec carries, or raise with every reason (no I/O)."""

    blockers = [*admission_blockers(manifest), *model_structure_blockers(manifest, structure),
                *station_reach_blockers(scene_plan, manifest)]
    if blockers:
        raise TeamEmbodimentError(blockers)
    binding = {"schema_version": BINDING_SCHEMA, "manifest": deepcopy(dict(manifest)),
               "model": {"sha256": model_record["sha256"], "size_bytes": int(model_record["size_bytes"]),
                         "bundle_path": BUNDLE_MODEL_PATH},
               "body_paths": list(structure["body_paths"]), "binding_digest": ""}
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
    structure = {"body_paths": list(binding["body_paths"])}
    value = deepcopy(dict(plan))
    station = (value.get("robot") or {}).get("base_pose_world")
    if not isinstance(station, Mapping):
        raise TeamEmbodimentError(["team_embodiment_station_unavailable"])
    gripper = manifest.get("gripper") or {}
    contact_links = list(gripper.get("contact_links") or [manifest["tool_link"]])
    task_bodies = sorted({_body_path(structure, link) for link in contact_links} - {None})
    if len(task_bodies) != len(contact_links):
        raise TeamEmbodimentError(["team_embodiment_contact_links_not_in_model"])
    reset = {name: float(position) for name, position in manifest["reset_joint_positions"].items()}
    value["robot"] = {
        "robot_id": ROBOT_ID, "embodiment_id": manifest["embodiment_id"],
        "embodiment_digest": manifest["embodiment_digest"], "form": manifest["form"],
        "base_mode": "fixed_at_station", "base_pose_world": deepcopy(dict(station)),
        "usd_path": BUNDLE_MODEL_PATH, "usd_sha256": binding["model"]["sha256"],
        "usd_size_bytes": int(binding["model"]["size_bytes"]), "usd_dependency_bindings": [],
        "usd_runtime_asset_dependencies": [],
        "arm_joint_names": [joint["name"] for joint in manifest["arm_joints"]],
        "joint_position_limits_rad": {joint["name"]: [float(joint["lower"]), float(joint["upper"])]
                                      for joint in manifest["arm_joints"]},
        "joint_velocity_limits": {joint["name"]: float(joint["max_velocity"]) for joint in manifest["arm_joints"]},
        "gripper_joints": [dict(joint) for joint in gripper.get("joints") or []],
        "gripper_max_width_m": gripper.get("max_width_m"),
        "joint_reset_positions_rad": reset, "actuator": dict(manifest["actuator"]),
        "tool_link": manifest["tool_link"], "camera_names": camera_names(manifest),
        "task_contact_body_paths": task_bodies,
        # The rigid readback's grasp position is the midpoint of two bodies; one contact link counts twice.
        "grasp_frame": {"kind": "body_midpoint", "body_names": (contact_links * 2)[:2]},
        "control_hz": int(manifest["control_hz"]), "action_chunk_rows": int(manifest["action_chunk_rows"]),
    }
    if isinstance(value.get("reset"), dict):
        value["reset"]["robot_joint_positions_rad"] = dict(reset)
    # The team's cameras replace Blueprint's policy cameras; the review (overview) camera stays.
    review = [row for row in value.get("cameras") or [] if row.get("review_only") is True]
    value["cameras"] = [*_plan_cameras(manifest, structure), *review]
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
        _rebind_contact_sensors(value["articulation"], task_bodies=task_bodies,
                                protected=sorted(binding["body_paths"]))
    profile = {"schema_version": "team_embodiment_profile.v1", "robot_preset_id": ROBOT_ID,
               "embodiment_id": manifest["embodiment_id"], "embodiment_digest": manifest["embodiment_digest"],
               "policy_interface_id": POLICY_INTERFACE_ID,
               "policy_camera_roles": [role for role, _ in camera_slots(manifest)],
               "review_only_camera_roles": [row.get("role") for row in review],
               "claim_boundary": {"base": "held_fixed_at_scene_robot_station",
                                  "task_sampling_hz": task_hz, "control_hz": hz},
               "binding_digest": binding["binding_digest"], "profile_digest": ""}
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    value["policy_canary_embodiment_profile"] = profile
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def is_team_plan(plan: Mapping[str, Any]) -> bool:
    return (plan.get("robot") or {}).get("robot_id") == ROBOT_ID


__all__ = [
    "ADMITTED_FORMS",
    "BINDING_SCHEMA",
    "BUNDLE_MODEL_PATH",
    "CONTROL_HZ",
    "FORMS",
    "POLICY_INTERFACE_ID",
    "ROBOT_ID",
    "SCHEMA",
    "TeamEmbodimentError",
    "action_dimension",
    "admission_blockers",
    "apply_team_embodiment",
    "camera_names",
    "camera_slots",
    "is_team_plan",
    "model_structure_blockers",
    "read_model_structure",
    "station_reach_blockers",
    "team_embodiment_binding",
    "validate_embodiment_manifest",
    "verify_model_file",
]
