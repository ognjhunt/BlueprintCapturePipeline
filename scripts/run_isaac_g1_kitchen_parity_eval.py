#!/usr/bin/env python3
"""Isaac Sim GPU runner: MuJoCo-parity G1 walk-to-target eval in the sim-ready Lightwheel kitchen.

Runs inside Isaac's python (``/isaac-sim/python.sh``) on the GPU worker. Self-contained: the
policy module ``isaac_g1_policy.py`` is shipped alongside this script in the bundle. Per
navigation scenario it drives the SAME deterministic walk-to-target controller as the MuJoCo
lane — proposing collision-checked candidate root poses, probing each via a PhysX overlap
query, kinematically placing the G1, and RTX-rendering overview + robot-POV frames into MP4 —
and records the MuJoCo-schema trace. Emits ``isaac_g1_kitchen_parity_result.json`` (same
task-outcome contract) + traces + MP4s and uploads the out dir via the provider signed-PUT.

Honesty boundary: Stage A is a *kinematic* navigation preview (parity with MuJoCo's preview
controller), RTX-rendered on Isaac. It is not dynamic locomotion and not a learned policy; the
GR00T N1.7 SONIC stage swaps the policy (``--policy groot_sonic``) without changing this harness.

The Isaac-API calls (boot, stage, PhysX overlap, Replicator render) are GPU-only and verified
on the worker, not locally; the non-Isaac helpers are unit-tested in the repo.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def _log(msg: str) -> None:
    """Flushed, timestamped progress line so the heartbeat-uploaded console shows exactly how
    far the runner got (Isaac ops between scene-load and render give no output otherwise)."""
    print(f"[parity {time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- policy import: bundle dir on the worker, package in the repo (tests) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import isaac_g1_policy as policy_mod  # bundle (worker)
except Exception:  # noqa: BLE001
    from blueprint_pipeline import isaac_g1_policy as policy_mod  # repo (tests)

RESULT_SCHEMA_VERSION = "isaac_g1_kitchen_parity_result.v1"
# robot footprint half-extent (m) for the PhysX overlap probe (approx G1 standing bbox)
ROBOT_FOOTPRINT_HALF_EXTENT = (0.28, 0.28, 0.62)
ROBOT_PELVIS_HEIGHT_M = 0.79
MANIPULATION_READY_ARM_SELECTIONS = ("right", "left", "both")
MANIPULATION_READY_ARM_JOINT_DELTAS = {
    "left": {
        "left_shoulder_pitch_joint": -0.85,
        "left_shoulder_roll_joint": 0.15,
        "left_shoulder_yaw_joint": 0.10,
        "left_elbow_joint": -0.23,
        "left_wrist_roll_joint": -0.10,
        "left_wrist_pitch_joint": -0.15,
    },
    "right": {
        "right_shoulder_pitch_joint": -0.85,
        "right_shoulder_roll_joint": -0.15,
        "right_shoulder_yaw_joint": -0.10,
        "right_elbow_joint": -0.23,
        "right_wrist_roll_joint": 0.10,
        "right_wrist_pitch_joint": -0.15,
    },
}


# ============================ testable helpers (no isaacsim) ============================

def load_request(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_scenarios(request: Mapping[str, Any]) -> list[dict]:
    """Normalize scenarios to {scenario_id, route_points:[[x,y,z],...], start, target, instruction}.
    Accepts explicit route_points, or spawn_position_xyz + target_position_xyz."""
    out: list[dict] = []
    for raw in request.get("scenarios", []) or []:
        sid = str(raw.get("scenario_id") or raw.get("id") or f"scenario_{len(out)+1}")
        route = raw.get("route_points") or raw.get("waypoints")
        start = raw.get("spawn_position_xyz") or raw.get("start") or (route[0] if route else None)
        target = raw.get("target_position_xyz") or raw.get("target") or (route[-1] if route else None)
        if start is None or target is None:
            continue
        start = [float(c) for c in start]
        target = [float(c) for c in target]
        if not route:
            route = [start, target]
        route = [[float(c) for c in p] for p in route]
        # lift the navigation route to pelvis height so the root trace is realistic
        route = [[p[0], p[1], ROBOT_PELVIS_HEIGHT_M] for p in route]
        out.append({
            "scenario_id": sid,
            "route_points": route,
            "start": [start[0], start[1], ROBOT_PELVIS_HEIGHT_M],
            "target": [target[0], target[1], ROBOT_PELVIS_HEIGHT_M],
            "instruction": str(raw.get("instruction") or raw.get("description") or ""),
            "scenario_eval_run_id": raw.get("scenario_eval_run_id"),
        })
    return out


def assemble_collision_summary(*, actions: Sequence[Mapping[str, Any]],
                               rejected_probe_total: int, response_event_total: int) -> dict:
    """Build the collision_summary that compute_task_outcome consumes from the per-step trace."""
    committed = sum(int(a.get("scene_collision_contact_count") or 0) for a in actions)
    return {
        "robot_scene_contact_event_count": committed,
        "rejected_scene_collision_probe_count": int(rejected_probe_total),
        "near_miss_event_count": int(rejected_probe_total),
        "collision_response_event_count": int(response_event_total),
        "clearance_threshold_m": policy_mod.TASK_CLEARANCE_THRESHOLD_M,
    }


def mp4_command(frames_glob: str, fps: int, out_path: str) -> list[str]:
    """ffmpeg command to assemble numbered PNG frames into an MP4 (yuv420p, web-playable)."""
    return ["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", frames_glob,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]


def build_result(*, scenarios: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]],
                 policy_id: str, kitchen_usd: str, g1_usd: str | None,
                 blockers: Sequence[str],
                 physics_articulation_contact_reports: Sequence[Mapping[str, Any]] | None = None) -> dict:
    passed = sum(1 for o in outcomes if o.get("task_success"))
    status = "completed" if outcomes and not blockers else "blocked"
    contact_summary = summarize_physics_articulation_contact_reports(
        physics_articulation_contact_reports or []
    )
    proof_boundary = (
        "Isaac RTX-rendered kinematic walk-to-target preview (parity with the MuJoCo preview "
        "controller). Not dynamic locomotion, not a learned policy, not deployment readiness."
    )
    if contact_summary["scenario_count"] > 0:
        proof_boundary = (
            "Isaac RTX-rendered kinematic walk-to-target preview plus opt-in PhysX articulation "
            "standing/contact settle samples. This upgrades the standing placement evidence to "
            "physics-stepped support/contact evidence, but it is still not full dynamic locomotion, "
            "not a learned balance controller, and not deployment readiness."
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "policy_id": policy_id,
        "kitchen_usd": kitchen_usd,
        "g1_usd": g1_usd,
        "scenario_count": len(scenarios),
        "scenarios_executed": len(outcomes),
        "scenarios_passed": passed,
        "rendered_by_isaac_rtx": True,
        "blockers": list(blockers),
        "scenarios": [
            {"scenario_id": s.get("scenario_id"), **o}
            for s, o in zip(scenarios, outcomes)
        ],
        "proof_boundary": proof_boundary,
    }
    if contact_summary["scenario_count"] > 0:
        result["physics_articulation_standing_contact_summary"] = contact_summary
        result["physics_articulation_standing_contact_reports"] = [
            dict(report) for report in physics_articulation_contact_reports or []
        ]
    return result


def summarize_physics_articulation_contact_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scenario_count = len(reports)
    completed = [r for r in reports if r.get("status") == "completed"]
    contact_records = sum(int(r.get("contact_event_count") or 0) for r in reports)
    support_records = sum(int(r.get("support_contact_event_count") or 0) for r in reports)
    return {
        "scenario_count": scenario_count,
        "completed_scenario_count": len(completed),
        "contact_event_count": contact_records,
        "support_contact_event_count": support_records,
        "all_completed": bool(scenario_count and len(completed) == scenario_count),
        "all_have_support_contact_evidence": bool(
            scenario_count and all(int(r.get("support_contact_event_count") or 0) > 0 for r in reports)
        ),
        "root_pose_teleport_during_physics_settle": any(
            bool(r.get("root_pose_teleport_during_physics_settle")) for r in reports
        ),
        "claim_boundary": (
            "PhysX articulation standing/contact settle evidence only. This does not prove full "
            "dynamic locomotion, learned balance, task manipulation success, or deployment readiness."
        ),
    }


def upload_zip(out_dir: Path, put_url: str | None) -> int | None:
    if not put_url:
        return None
    import urllib.request
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out_dir).as_posix())
    req = urllib.request.Request(put_url, data=buf.getvalue(), method="PUT",
                                 headers={"Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return int(getattr(r, "status", 200))


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about +Z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _norm(v):
    m = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """USD-camera look-at orientation as (w, x, y, z). The camera views along its local -Z with
    +Y up; we build the basis [x, y, z] with z = -forward and convert to a quaternion."""
    forward = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    zc = (-forward[0], -forward[1], -forward[2])            # camera local +Z (out of screen)
    xc = _norm(_cross(up, zc))
    if xc == (0.0, 0.0, 0.0):                               # up parallel to view dir
        xc = _norm(_cross((0.0, 1.0, 0.0), zc))
    yc = _cross(zc, xc)
    m00, m01, m02 = xc[0], yc[0], zc[0]
    m10, m11, m12 = xc[1], yc[1], zc[1]
    m20, m21, m22 = xc[2], yc[2], zc[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def project_point_to_pixel(world_pt, eye, target, up, vfov_deg: float, width: int, height: int):
    """Pinhole-project a world point into the camera image. Returns (u, v, depth) in pixels if the
    point is in front of the camera and within frame, else None. Used to build the G1 skeleton
    landmarks (joint world positions -> 2D image landmarks) for OSCAR conditioning."""
    fwd = _norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    right = _norm(_cross(fwd, up))
    if right == (0.0, 0.0, 0.0):
        right = _norm(_cross(fwd, (0.0, 1.0, 0.0)))
    tup = _cross(right, fwd)
    rel = (world_pt[0] - eye[0], world_pt[1] - eye[1], world_pt[2] - eye[2])
    z = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2]
    if z <= 1e-6:
        return None
    x = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]
    y = rel[0] * tup[0] + rel[1] * tup[1] + rel[2] * tup[2]
    f = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    u = width / 2.0 + f * (x / z)
    v = height / 2.0 - f * (y / z)
    if 0.0 <= u < width and 0.0 <= v < height:
        return (u, v, z)
    return None


def scene_framing(scenarios: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float], float]:
    """Center + radius of all scenario route points, for the static overview camera."""
    pts = [p for sc in scenarios for p in sc.get("route_points", [])]
    if not pts:
        return (0.0, 0.0, ROBOT_PELVIS_HEIGHT_M), 4.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    radius = max((math.hypot(p[0] - cx, p[1] - cy) for p in pts), default=2.0)
    return (cx, cy, ROBOT_PELVIS_HEIGHT_M), max(2.5, radius)


def follow_cam_pose(root_pose, yaw, *, back: float = 2.2, up: float = 1.6):
    """Eye + target for a robot-POV follow camera: behind and above the root, looking ahead."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    eye = (root_pose[0] - fx * back, root_pose[1] - fy * back, root_pose[2] + up)
    target = (root_pose[0] + fx * 1.5, root_pose[1] + fy * 1.5, root_pose[2] + 0.2)
    return eye, target


def verify_cam_pose(root_pose, yaw, *, back: float = 2.4, up: float = 1.5, side: float = 1.2):
    """3rd-person VERIFICATION camera: pulled back behind + above + to the side so the WHOLE robot AND
    the workspace it faces are both in frame — proves where the robot is actually standing (vs the
    egocentric POV, which shows only what the robot looks at)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    px, py = -fy, fx  # perpendicular (left of facing) for a 3/4 angle that reveals body-vs-counter gap
    eye = (root_pose[0] - fx * back + px * side, root_pose[1] - fy * back + py * side, root_pose[2] + up)
    target = (root_pose[0] + fx * 0.45, root_pose[1] + fy * 0.45, root_pose[2] + 0.25)  # robot torso/front
    return eye, target


def manipulation_cam_pose(root_pose, yaw, *, eye_forward: float = 0.15, eye_height: float = 1.35,
                          target_forward: float = 0.6, target_height: float = 0.9, look_at=None):
    """Eye + target for an EGOCENTRIC manipulation POV: from the robot's head, looking down-forward
    at the workspace directly in front (the sink/faucet and the robot's hands).

    Unlike ``follow_cam_pose`` (a chase shot behind+above, framing the whole robot walking across the
    room) this frames the local task region. Heights are absolute so the view sits at head level and
    looks at counter level — the in-distribution, coherent view a manipulation WAM can actually
    predict, instead of a room-scale navigation scene it collapses to blur on.

    ``look_at`` (a fixed world x,y,z — e.g. the faucet's known position) pins the target so the
    workspace stays centered regardless of the policy's noisy final yaw; without it the target is
    derived yaw-relative (forward of the robot)."""
    fx, fy = math.cos(yaw), math.sin(yaw)
    eye = (root_pose[0] + fx * eye_forward, root_pose[1] + fy * eye_forward, eye_height)
    if look_at is not None:
        target = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
    else:
        target = (root_pose[0] + fx * target_forward, root_pose[1] + fy * target_forward, target_height)
    return eye, target


# ============================ Isaac-only (GPU worker) ============================

def _boot_sim(headless: bool = True):
    from isaacsim import SimulationApp  # type: ignore
    return SimulationApp({"headless": headless, "renderer": "RayTracedLighting"})


def _extension_toggle():
    try:
        from isaacsim.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    except Exception:  # noqa: BLE001
        from omni.isaac.core.utils.extensions import enable_extension, disable_extension  # type: ignore
    return enable_extension, disable_extension


def _enable_and_import_replicator():
    """Enable the Replicator extension (needed for render products) and import it. Must be
    called AFTER SimulationApp boots — omni.* modules are not importable before Kit starts."""
    enable_extension, _ = _extension_toggle()
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep  # type: ignore
    return rep


def _disable_physics_cooking() -> None:
    """Disable ONLY the PhysX collision-*cooking* extension (not the physx core, which the RTX
    renderer depends on), so the 47-object kitchen's SDF/convex cooking can't block the render.
    Also push every collision approximation to the cheapest box via carb settings, so any residual
    cooking is trivial. The kinematic preview needs no kitchen physics anyway."""
    _, disable_extension = _extension_toggle()
    try:
        disable_extension("omni.physx.cooking")
    except Exception:  # noqa: BLE001
        pass
    try:
        import carb  # type: ignore
        s = carb.settings.get_settings()
        s.set_bool("/physics/cooking/ujitsoCollisionCooking", False)
        s.set_bool("/persistent/physics/visualizationDisplayColliders", False)
        s.set_bool("/physics/collisionConeCustomGeometry", False)
    except Exception:  # noqa: BLE001
        pass


def _open_stage(usd_path: str):
    import omni.usd  # type: ignore
    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    return ctx.get_stage()


def _resolve_asset_uri(value: str) -> str:
    """Resolve a relative Isaac asset path (e.g. 'Isaac/Robots/Unitree/G1/g1.usd') against the
    Isaac assets root on the worker. Absolute paths / URIs pass through unchanged."""
    if "://" in value or value.startswith("/") or value.startswith("omniverse:"):
        return value
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore
        root = get_assets_root_path()
        if root:
            return root.rstrip("/") + "/" + value.lstrip("/")
    except Exception:  # noqa: BLE001
        pass
    return value


def _bind_g1(stage, g1_usd: str, prim_path: str = "/World/G1"):
    """Reference the official Isaac G1 USD and verify it is a controllable, collidable articulation."""
    from pxr import UsdPhysics  # type: ignore
    g1_prim = stage.DefinePrim(prim_path, "Xform")
    g1_prim.GetReferences().AddReference(g1_usd)
    art_count = collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
    return {
        "prim_path": prim_path,
        "controllable_articulation_detected": art_count > 0,
        "collision_enabled_verified": collision_count > 0,
        "articulation_root_api_prim_count": art_count,
        "collision_api_prim_count": collision_count,
    }


def _setup_g1_articulation(prim_path: str):
    """Create + initialize an Isaac Articulation on the bound G1 so we can drive its joints
    (the procedural walk gait) and read its link world poses (the skeleton). Returns
    (articulation, dof_index_by_name, default_joint_positions, link_names). GPU-only."""
    from isaacsim.core.prims import SingleArticulation  # type: ignore
    art = SingleArticulation(prim_path=prim_path, name="g1")
    art.initialize()
    dof_names = list(art.dof_names or [])
    dof_index = {n: i for i, n in enumerate(dof_names)}
    import numpy as np  # type: ignore
    default = np.asarray(art.get_joint_positions()).astype("float32")
    link_names = list(getattr(art, "body_names", []) or [])
    return art, dof_index, default, link_names


def manipulation_ready_arm_joint_deltas(arm: str = "both") -> dict[str, float]:
    """Joint deltas that raise G1 forearms into a first-person manipulation-ready pose.

    The values are relative to the standing keyframe, so the pose is portable across Isaac and
    MuJoCo G1 assets without hard-coding absolute default qpos values.
    """
    selection = str(arm or "both").strip().lower()
    if selection not in MANIPULATION_READY_ARM_SELECTIONS:
        raise ValueError(f"unknown manipulation arm selection: {arm!r}")
    sides = ("left", "right") if selection == "both" else (selection,)
    out: dict[str, float] = {}
    for side in sides:
        out.update(MANIPULATION_READY_ARM_JOINT_DELTAS[side])
    return out


def _apply_joint_deltas(targets, default, dof_index, deltas: Mapping[str, float]) -> list[str]:
    applied: list[str] = []
    for name, delta in deltas.items():
        idx = dof_index.get(name)
        if idx is not None and idx < len(targets) and idx < len(default):
            targets[idx] = default[idx] + float(delta)
            applied.append(name)
    return applied


def _joint_targets_for_pose(
    default,
    dof_index,
    *,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
):
    import numpy as np  # type: ignore
    targets = np.array(default, dtype="float32", copy=True)
    _apply_joint_deltas(targets, default, dof_index, policy_mod.gait_joint_deltas(phase, moving))
    if manipulation_ready:
        _apply_joint_deltas(
            targets,
            default,
            dof_index,
            manipulation_ready_arm_joint_deltas(manipulation_reach_arm),
        )
    return targets


def _apply_articulation_joint_targets(art, targets):
    """Prefer Isaac's articulation action path, falling back to direct joint state writes on older
    worker images. The return value is persisted in the contact report for auditability."""
    try:
        from isaacsim.core.utils.types import ArticulationAction  # type: ignore
        art.apply_action(ArticulationAction(joint_positions=targets))
        return "articulation_action_position_targets"
    except Exception:  # noqa: BLE001
        art.set_joint_positions(targets)
        return "direct_joint_state_position_set"


def _drive_g1_walk(
    art,
    dof_index,
    default,
    *,
    root_pose,
    yaw,
    phase,
    moving,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
):
    """Set the G1 root world pose + joint positions = standing + gait deltas (kinematic pose)."""
    import numpy as np  # type: ignore
    w, x, y, z = yaw_to_quat(float(yaw))
    art.set_world_pose(position=np.asarray(root_pose, dtype="float32"),
                       orientation=np.asarray([w, x, y, z], dtype="float32"))
    targets = _joint_targets_for_pose(
        default,
        dof_index,
        phase=phase,
        moving=moving,
        manipulation_ready=manipulation_ready,
        manipulation_reach_arm=manipulation_reach_arm,
    )
    art.set_joint_positions(targets)
    return targets


def _g1_skeleton_world_positions(art, link_names):
    """World-space positions of the G1 links (the skeleton landmarks before projection)."""
    import numpy as np  # type: ignore
    positions, _ = art.get_link_world_poses()
    positions = np.asarray(positions)
    return [(link_names[i] if i < len(link_names) else f"link_{i}",
             (float(positions[i][0]), float(positions[i][1]), float(positions[i][2])))
            for i in range(len(positions))]


def _project_skeleton(skeleton_world, *, eye, target, up, vfov_deg, width, height):
    """Project G1 link world positions into the camera -> OSCAR-schema landmark list. Each landmark
    is {landmark_id, image_projection:{available,u_px,v_px,depth_m}} (the exact shape the OSCAR WAM
    input-package materialization reads)."""
    landmarks = []
    for name, wp in skeleton_world:
        px = project_point_to_pixel(wp, eye, target, up, vfov_deg, width, height)
        if px is not None:
            landmarks.append({"landmark_id": name, "image_projection": {
                "available": True, "u_px": round(px[0], 2), "v_px": round(px[1], 2),
                "depth_m": round(px[2], 4)}})
    return landmarks


def _g1_link_rest_offsets(stage, prim_path: str):
    """Pure-USD G1 skeleton: rest-pose offset (in the root frame) of each link prim under the G1.
    No physics/tensor-view (which gets invalidated on this G1 USD) — just the link transforms.
    Returns [(name, (dx,dy,dz)), ...]. Per-step world = root_pose + Rz(yaw) @ offset."""
    from pxr import Usd, UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    root_prim = stage.GetPrimAtPath(prim_path)
    rt = xc.GetLocalToWorldTransform(root_prim).ExtractTranslation()
    root = (float(rt[0]), float(rt[1]), float(rt[2]))
    offs = []
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if "link" not in name.lower() or not prim.IsA(UsdGeom.Xformable):
            continue
        t = xc.GetLocalToWorldTransform(prim).ExtractTranslation()
        offs.append((name, (float(t[0]) - root[0], float(t[1]) - root[1], float(t[2]) - root[2])))
    return offs


def _rest_skeleton_world(offsets, root_pose, yaw):
    """Place the rest-pose link offsets at the robot's per-step root pose (translate + Z-rotate)."""
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    out = []
    for name, (ox, oy, oz) in offsets:
        out.append((name, (root_pose[0] + cy * ox - sy * oy,
                           root_pose[1] + sy * ox + cy * oy,
                           root_pose[2] + oz)))
    return out


def skeleton_world_for_frame(*, art_ctx, rest_offsets, root_pose, yaw):
    """Return the best available G1 skeleton for a rendered frame.

    Some Isaac worker images expose a controllable G1 articulation with valid joints but no body
    names. Reading link poses in that state can invalidate the PhysX tensor view, so fall back to
    the USD rest-offset skeleton unless the articulation has usable link names.
    """
    if art_ctx is not None and art_ctx.get("link_names"):
        try:
            return _g1_skeleton_world_positions(art_ctx["art"], art_ctx["link_names"])
        except Exception as exc:  # noqa: BLE001
            _log(f"G1 articulation skeleton read failed ({exc!r}); using USD skeleton fallback")
    if rest_offsets is not None:
        return _rest_skeleton_world(rest_offsets, root_pose, yaw)
    return []


def compute_arm_reach_skeleton(skeleton, target, reach_frac, *, arm: str = "right"):
    """Re-pose one arm of a world-space skeleton so its hand reaches toward ``target`` (the faucet).

    The walk policy never moves the arms, so the skeleton (OSCAR's action conditioning) just shows a
    rigid robot. This rotates the arm chain about the shoulder so the hand travels from its rest spot
    to the target as ``reach_frac`` goes 0->1 — turning the skeleton-video into an actual reach. Each
    arm link keeps its rest fractional distance from the shoulder (rigid straight-arm reach), and the
    reach is clamped to the arm's length so it never overstretches. Pure geometry, GPU-independent.

    ``skeleton`` is ``[(name, (x,y,z)), ...]``; returns the same shape with the arm links re-placed.
    """
    if target is None or reach_frac <= 0.0:
        return skeleton
    if str(arm).lower() == "both":
        out = skeleton
        for side in ("left", "right"):
            out = compute_arm_reach_skeleton(out, target, reach_frac, arm=side)
        return out
    arm_keys = ("shoulder", "elbow", "wrist", "hand")
    prefix = f"{arm}_"
    arm_pts = [(n, p) for n, p in skeleton if n.startswith(prefix) and any(k in n for k in arm_keys)]
    sh = [p for n, p in arm_pts if "shoulder" in n]
    hand = [p for n, p in arm_pts if "hand" in n]
    if not sh or not hand:
        return skeleton

    def centroid(ps):
        return tuple(sum(c) / len(ps) for c in zip(*ps))

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def scale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def length(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    shoulder = centroid(sh)
    hand_rest = centroid(hand)
    arm_len = length(sub(hand_rest, shoulder)) or 1e-6
    to_target = sub(target, shoulder)
    tlen = length(to_target) or 1e-6
    reach_dist = min(arm_len, tlen)
    hand_reach = add(shoulder, scale(to_target, reach_dist / tlen))  # clamped along shoulder->target
    frac = max(0.0, min(1.0, float(reach_frac)))
    hand_now = add(scale(hand_rest, 1.0 - frac), scale(hand_reach, frac))
    out = []
    for n, p in skeleton:
        if n.startswith(prefix) and any(k in n for k in arm_keys):
            f = length(sub(p, shoulder)) / arm_len  # rest fractional distance along the arm
            out.append((n, add(shoulder, scale(sub(hand_now, shoulder), f))))
        else:
            out.append((n, p))
    return out


def arm_reach_rotation(shoulder, rest_elbow, target, reach_frac):
    """Axis (unit xyz) + angle (radians) of the kinematic SHOULDER rotation that swings the rest
    upper-arm bone (shoulder->rest_elbow) toward the target (shoulder->target), scaled by reach_frac.

    Axis-agnostic: it derives the rotation from the rest bone and the desired bone direction, so there
    is NO hardcoded joint axis to inspect on the G1 USD. Applied about the shoulder pivot it points the
    upper arm at the object; the elbow/wrist/hand follow rigidly. Pure geometry, GPU-independent."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def length(a):
        return math.sqrt(dot(a, a))

    def norm(a):
        L = length(a) or 1e-9
        return (a[0] / L, a[1] / L, a[2] / L)

    rest = norm(sub(rest_elbow, shoulder))
    want = norm(sub(target, shoulder))
    d = max(-1.0, min(1.0, dot(rest, want)))
    angle = math.acos(d) * max(0.0, min(1.0, float(reach_frac)))
    axis = cross(rest, want)
    if length(axis) < 1e-6:
        axis = (0.0, 0.0, 1.0)  # rest ~parallel/antiparallel to want -> arbitrary axis (angle ~0/pi)
    return norm(axis), angle


def _find_arm_link(links: dict, *keys: str):
    """First Xformable link prim whose name contains ALL keys (case-insensitive)."""
    for name, prim in links.items():
        low = name.lower()
        if all(k.lower() in low for k in keys):
            return prim
    return None


def _pose_arm_kinematic_usd(stage, prim_path: str, target, *, arm: str = "right",
                            reach_frac: float = 1.0) -> int:
    """Kinematically pose the G1 arm(s) so the upper arm points at ``target`` — pure USD (rotate the
    shoulder link about its pivot, children follow), NO physics tensor view, so it cannot trigger the
    crash the articulation drive does. Returns the number of arms posed. GPU/USD only."""
    from pxr import Usd, UsdGeom, Gf  # type: ignore
    sides = ("left", "right") if arm == "both" else (arm,)
    root = stage.GetPrimAtPath(prim_path)
    links = {p.GetName(): p for p in Usd.PrimRange(root)
             if p.IsA(UsdGeom.Xformable) and "link" in p.GetName().lower()}
    posed = 0
    for side in sides:
        shoulder = (_find_arm_link(links, side, "shoulder", "pitch")
                    or _find_arm_link(links, side, "shoulder"))
        elbow = _find_arm_link(links, side, "elbow")
        if shoulder is None or elbow is None:
            continue
        xc = UsdGeom.XformCache()  # fresh cache per arm (previous arm's mutation invalidated it)
        sh_w = xc.GetLocalToWorldTransform(shoulder)
        el_w = xc.GetLocalToWorldTransform(elbow)
        sp = sh_w.ExtractTranslation()
        ep = el_w.ExtractTranslation()
        axis, angle = arm_reach_rotation((sp[0], sp[1], sp[2]), (ep[0], ep[1], ep[2]),
                                         (float(target[0]), float(target[1]), float(target[2])),
                                         reach_frac)
        if angle < 1e-4:
            continue
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis), math.degrees(angle)))
        pivot = Gf.Vec3d(sp[0], sp[1], sp[2])
        # rotate the shoulder's world transform about the shoulder pivot (USD row-vector convention)
        m_pivot = Gf.Matrix4d().SetTranslate(-pivot) * rot * Gf.Matrix4d().SetTranslate(pivot)
        new_world = sh_w * m_pivot
        parent_world = xc.GetLocalToWorldTransform(shoulder.GetParent())
        new_local = new_world * parent_world.GetInverse()
        xf = UsdGeom.Xformable(shoulder)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(new_local)
        posed += 1
    return posed


def _setup_articulated_g1(prim_path: str, *, gravity_z: float = 0.0):
    """Create a physics SimulationContext (gravity OFF, so the kinematic walk pose holds without the
    G1 collapsing), play it, and initialize the G1 articulation for joint driving + link readback.
    Returns a context dict. GPU-only."""
    from isaacsim.core.api import SimulationContext  # type: ignore
    ctx = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    ctx.initialize_physics()
    try:
        ctx.get_physics_context().set_gravity(float(gravity_z))
    except Exception:  # noqa: BLE001
        pass
    art, dof_index, default, link_names = _setup_g1_articulation(prim_path)
    ctx.play()
    return {"ctx": ctx, "art": art, "dof_index": dof_index, "default": default,
            "link_names": link_names, "dof_count": len(dof_index), "gravity_z": float(gravity_z)}


def _sim_step(ctx, *, render: bool = False) -> None:
    try:
        ctx.step(render=render)
    except TypeError:
        ctx.step()


def _safe_articulation_world_pose(art) -> dict[str, Any]:
    try:
        pos, quat = art.get_world_pose()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    try:
        return {
            "available": True,
            "position_xyz": [round(float(v), 6) for v in pos],
            "orientation_wxyz": [round(float(v), 6) for v in quat],
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}


def _path_from_encoded_sdf(value) -> str:
    try:
        from pxr import PhysicsSchemaTools  # type: ignore
        return str(PhysicsSchemaTools.intToSdfPath(int(value)))
    except Exception:  # noqa: BLE001
        return str(value)


def _vec3_to_list(value) -> list[float] | None:
    try:
        return [round(float(value[i]), 6) for i in range(3)]
    except Exception:  # noqa: BLE001
        try:
            return [round(float(getattr(value, axis)), 6) for axis in ("x", "y", "z")]
        except Exception:  # noqa: BLE001
            return None


def _enable_contact_reports(stage, robot_prim_path: str, *, threshold: float = 0.0) -> dict[str, Any]:
    """Apply PhysX contact-report API to the articulation/root and likely foot links.

    This is best-effort because Isaac worker images and G1 USD variants differ. A failure should
    block only the contact report, not the render path.
    """
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": repr(exc), "enabled_paths": []}
    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        return {"status": "unavailable", "error": "robot_prim_not_found", "enabled_paths": []}
    enabled: list[str] = []
    candidates = []
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if (
            prim.GetPath() == root.GetPath()
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or ("foot" in name and prim.HasAPI(UsdPhysics.CollisionAPI))
        ):
            candidates.append(prim)
    for prim in candidates:
        try:
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(float(threshold))
            enabled.append(str(prim.GetPath()))
        except Exception:  # noqa: BLE001
            continue
    return {"status": "enabled" if enabled else "unavailable", "enabled_paths": enabled}


def _contact_report_records(robot_prim_path: str, *, max_records: int = 40) -> list[dict[str, Any]]:
    try:
        from omni.physx import get_physx_simulation_interface  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    try:
        report = get_physx_simulation_interface().get_contact_report()
    except Exception:  # noqa: BLE001
        return []
    if not report:
        return []
    try:
        headers, data = report[0], report[1] if len(report) > 1 else []
    except Exception:  # noqa: BLE001
        return []
    records: list[dict[str, Any]] = []
    for header in list(headers)[:max_records]:
        actor0 = _path_from_encoded_sdf(getattr(header, "actor0", ""))
        actor1 = _path_from_encoded_sdf(getattr(header, "actor1", ""))
        collider0 = _path_from_encoded_sdf(getattr(header, "collider0", actor0))
        collider1 = _path_from_encoded_sdf(getattr(header, "collider1", actor1))
        joined = " ".join((actor0, actor1, collider0, collider1)).lower()
        if robot_prim_path.lower() not in joined and "/world/g1" not in joined:
            continue
        offset = int(getattr(header, "contact_data_offset", 0) or 0)
        count = int(getattr(header, "num_contact_data", 0) or 0)
        samples = []
        for sample in list(data)[offset: offset + min(count, 3)]:
            samples.append({
                "position_xyz": _vec3_to_list(getattr(sample, "position", None)),
                "normal_xyz": _vec3_to_list(getattr(sample, "normal", None)),
                "impulse": (
                    round(float(getattr(sample, "impulse", 0.0) or 0.0), 6)
                    if hasattr(sample, "impulse") else None
                ),
            })
        records.append({
            "actor0": actor0,
            "actor1": actor1,
            "collider0": collider0,
            "collider1": collider1,
            "contact_data_count": count,
            "samples": samples,
        })
    return records


def _is_support_contact(record: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key) or "").lower()
        for key in ("actor0", "actor1", "collider0", "collider1")
    )
    return ("foot" in text or "ankle" in text or "toe" in text) and (
        "floor" in text or "ground" in text or "room" in text or "kitchen" in text
    )


def _settle_dynamic_standing_contacts(
    *,
    stage,
    art_ctx,
    robot_prim_path: str,
    root_pose,
    yaw,
    phase,
    moving,
    settle_steps: int,
    scenario_id: str,
    manipulation_ready: bool = False,
    manipulation_reach_arm: str = "both",
) -> dict[str, Any]:
    """Run a bounded PhysX standing/contact settle without mutating the G1 USD xform after the
    articulation tensor view exists.

    The policy route remains kinematic. This mode upgrades each sampled placement by stepping the
    real articulation against the scene with gravity and contact reporting; it is not a full dynamic
    walking controller.
    """
    import numpy as np  # type: ignore

    art = art_ctx["art"]
    ctx = art_ctx["ctx"]
    targets = _joint_targets_for_pose(
        art_ctx["default"],
        art_ctx["dof_index"],
        phase=phase,
        moving=moving,
        manipulation_ready=manipulation_ready,
        manipulation_reach_arm=manipulation_reach_arm,
    )
    w, x, y, z = yaw_to_quat(float(yaw))
    art.set_world_pose(
        position=np.asarray(root_pose, dtype="float32"),
        orientation=np.asarray([w, x, y, z], dtype="float32"),
    )
    command_mode = _apply_articulation_joint_targets(art, targets)
    before = _safe_articulation_world_pose(art)
    contact_setup = _enable_contact_reports(stage, robot_prim_path)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    executed = 0
    for _ in range(max(0, int(settle_steps))):
        try:
            _apply_articulation_joint_targets(art, targets)
            _sim_step(ctx, render=False)
            executed += 1
            if len(records) < 80:
                records.extend(_contact_report_records(robot_prim_path, max_records=20))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
            break
    after = _safe_articulation_world_pose(art)
    support_records = [r for r in records if _is_support_contact(r)]
    return {
        "schema_version": "isaac_g1_physics_articulation_standing_contact_report.v1",
        "status": "completed" if executed == max(0, int(settle_steps)) and not errors else "blocked",
        "scenario_id": scenario_id,
        "gravity_z": art_ctx.get("gravity_z"),
        "requested_settle_steps": int(settle_steps),
        "executed_settle_steps": executed,
        "root_pose_seeded_once_before_settle": True,
        "root_pose_teleport_during_physics_settle": False,
        "usd_root_xform_mutated_after_tensor_view": False,
        "joint_command_mode": command_mode,
        "contact_report_setup": contact_setup,
        "contact_event_count": len(records),
        "support_contact_event_count": len(support_records),
        "sample_contact_records": records[:20],
        "root_pose_before_settle": before,
        "root_pose_after_settle": after,
        "errors": errors,
        "claim_boundary": (
            "Physics articulation standing/contact settle for this sampled placement only; not "
            "full dynamic walking, learned balance control, task success, safety validation, or "
            "deployment readiness."
        ),
    }


def _overlap_probe(robot_prim_path: str, ground_prim_path: str = "/World/GroundPlane"):
    """Return probe(pose, yaw) -> scene-collision hit count using a PhysX box overlap of the
    robot footprint at the candidate pose, excluding the robot's own prims and the ground."""
    from omni.physx import get_physx_scene_query_interface  # type: ignore
    import carb  # type: ignore

    sqi = get_physx_scene_query_interface()
    hx, hy, hz = ROBOT_FOOTPRINT_HALF_EXTENT

    def probe(pose, yaw) -> int:
        hits = {"n": 0}

        def report(hit):  # noqa: ANN001
            path = str(getattr(hit, "collision", "") or getattr(hit, "rigid_body", ""))
            if not path.startswith(robot_prim_path) and not path.startswith(ground_prim_path):
                hits["n"] += 1
            return True  # keep scanning

        w, x, y, z = yaw_to_quat(float(yaw))
        sqi.overlap_box(
            carb.Float3(hx, hy, hz),
            carb.Float3(float(pose[0]), float(pose[1]), float(pose[2])),
            carb.Float4(x, y, z, w),  # PhysX quat order is (x,y,z,w)
            report, False,
        )
        return hits["n"]

    return probe


def _place_root(stage, prim_path: str, pose, yaw) -> None:
    from pxr import UsdGeom, Gf  # type: ignore
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    xform.AddRotateZOp().Set(math.degrees(float(yaw)))


def _place_camera(stage, cam_path: str, eye, target) -> None:
    from pxr import UsdGeom, Gf  # type: ignore
    w, x, y, z = look_at_quat(eye, target)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
    xform.AddOrientOp().Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def _force_cheap_collision(stage, approximation: str = "boundingCube") -> int:
    """Override every mesh collision approximation. The 47-object kitchen's default SDF cooking on
    non-watertight meshes takes >4 min and blocks the RTX render; ``boundingCube`` cooks ~instantly
    but is coarse (collision volumes far bigger than the visual shape, which shoves the robot off a
    head-on approach). ``convexHull`` is shape-accurate enough for the robot to stand centered + close
    and still cooks far faster than SDF (a watertight convex per mesh). Visual geometry is untouched.
    Returns the number of meshes overridden."""
    from pxr import UsdPhysics  # type: ignore
    tokens = {
        "boundingCube": UsdPhysics.Tokens.boundingCube,
        "convexHull": UsdPhysics.Tokens.convexHull,
        "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    }
    approx = tokens.get(approximation, UsdPhysics.Tokens.boundingCube)
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set(approx)
            n += 1
    return n


def _prune_to_focus(stage, route_points, focus_radius: float, keep_substrings) -> dict:
    """Task-aware scene subset: deactivate kitchen object prims whose placement is farther than
    ``focus_radius`` (m, xy) from the robot's route, keeping the task region + structural shell
    (walls/floor/lights). Deactivated prims are excluded from BOTH PhysX cooking and the render,
    so this is the lever for 'the scene is too large'. Returns {kept, pruned, kept_names}."""
    from pxr import UsdGeom  # type: ignore
    xc = UsdGeom.XformCache()
    keep_subs = [s.strip().lower() for s in keep_substrings if s and s.strip()]
    pts = [(float(p[0]), float(p[1])) for p in route_points] or [(0.0, 0.0)]
    root = stage.GetPrimAtPath("/root")
    if not (root and root.IsValid()):
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    kept, pruned, kept_names = 0, 0, []
    for child in root.GetChildren():
        name = child.GetName()
        low = name.lower()
        if any(s in low for s in keep_subs):
            kept += 1
            kept_names.append(name)
            continue
        try:
            t = xc.GetLocalToWorldTransform(child).ExtractTranslation()
            pos = (float(t[0]), float(t[1]))
            dmin = min(math.hypot(pos[0] - x, pos[1] - y) for x, y in pts)
        except Exception:  # noqa: BLE001
            kept += 1
            kept_names.append(name)
            continue
        if dmin <= focus_radius:
            kept += 1
            kept_names.append(name)
        else:
            child.SetActive(False)
            pruned += 1
    return {"kept": kept, "pruned": pruned, "kept_names": kept_names[:40]}


def _make_render_product(camera_path: str, width: int, height: int):
    import omni.replicator.core as rep  # type: ignore
    rp = rep.create.render_product(camera_path, (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return annot


def _save_rgb(annot, out_path: Path) -> bool:
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    data = annot.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        return False
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    Image.fromarray(arr.astype("uint8")).save(out_path)
    return True


def camera_aperture_for_fov(vfov_deg: float, width: int, height: int, focal_mm: float = 20.0):
    """Focal length + (horizontal, vertical) aperture that give a camera a vertical FOV of
    ``vfov_deg`` at the render aspect ratio. USD's default 50mm/20.955mm camera is a ~24deg
    telephoto — far too zoomed for the manipulation POV (it fills the frame with the dark sink
    basin) and it does NOT match the FOV the skeleton projection assumes, so the projected
    landmarks misalign with the render. Pure trig (no USD) so it is unit-testable."""
    vap = 2.0 * float(focal_mm) * math.tan(math.radians(float(vfov_deg)) / 2.0)
    hap = vap * (float(width) / float(height))
    return float(focal_mm), hap, vap


def _set_camera_fov(stage, cam_path: str, vfov_deg: float, width: int, height: int) -> None:
    """Set a USD camera's focal length + apertures so its vertical FOV == ``vfov_deg`` (matching the
    skeleton-projection FOV) instead of the narrow ~17deg default. GPU/USD only."""
    from pxr import UsdGeom  # type: ignore
    focal, hap, vap = camera_aperture_for_fov(vfov_deg, width, height)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(hap)
    cam.GetVerticalApertureAttr().Set(vap)


def _add_workspace_fill_light(stage, target, *, intensity: float, height: float = 2.0,
                              path: str = "/World/WorkspaceFill") -> None:
    """Add a local sphere fill light above the manipulation workspace (the faucet) so the dark sink
    basin + the reaching arm are lit. Intensity is configurable (blind-tunable via re-render). The
    default scene has a single distant key light that leaves the basin interior in shadow. GPU/USD."""
    from pxr import UsdLux, UsdGeom, Gf  # type: ignore
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    light.CreateRadiusAttr(0.4)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(float(target[0]), float(target[1]) - 0.25, float(height)))


def _neutralize_environment(stage, *, intensity: float = 1500.0) -> int:
    """Replace any outdoor-HDRI DomeLight in the loaded scene with a NEUTRAL uniform environment.

    The Lightwheel kitchen ships a DomeLight (e.g. ``DomeLight_01`` with ``texture/chive 7000x3500.hdr``)
    that projects an outdoor cityscape, visible through the kitchen windows — an incongruous background
    for an enclosed manipulation scene. Clearing the HDRI texture + setting a neutral bright color turns
    it into even ambient: windows read neutral (no city) AND the dark cabinet/basin surfaces get lifted
    by global fill. Returns the number of dome lights neutralized. GPU/USD only."""
    from pxr import UsdLux, Sdf, Gf  # type: ignore
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "DomeLight":
            continue
        dome = UsdLux.DomeLight(prim)
        for attr_name in ("inputs:texture:file", "texture:file"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                try:
                    attr.Set(Sdf.AssetPath(""))  # drop the HDRI -> uniform dome (no cityscape)
                except Exception:  # noqa: BLE001
                    pass
        try:
            dome.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.92, 0.95))  # neutral cool-white
            dome.CreateIntensityAttr(float(intensity))
        except Exception:  # noqa: BLE001
            pass
        n += 1
    return n


def run_scenarios(*, kitchen_usd: str, g1_usd: str, scenarios: Sequence[dict], out_dir: Path,
                  policy_id: str, steps: int, width: int, height: int, fps: int,
                  warmup_frames: int, capture_every: int, no_collision_probe: bool = False,
                  per_scenario_seconds: int = 480, focus_radius: float = 0.0,
                  keep_substrings: Sequence[str] = ("room", "floor", "wall", "ground", "ceiling", "light"),
                  disable_physx: bool = False, settle_seconds: int = 0,
                  cheap_collision: bool = False, articulated: bool = False,
                  camera_vfov_deg: float = 50.0, manipulation_cam: bool = False,
                  manipulation_look_at=None, render_subframes: int = 1,
                  manipulation_reach: bool = False, manipulation_reach_arm: str = "both",
                  fill_light_intensity: float = 0.0,
                  physics_articulation_drive: bool = False,
                  dynamic_standing_contact_steps: int = 0,
                  neutral_environment: bool = False,
                  kinematic_arm_pose: bool = False,
                  collision_approximation: str = "boundingCube",
                  verify_cam: bool = False) -> dict:
    """GPU orchestration: boot Isaac, load scene + G1, run the controller per scenario with RTX
    render + (optional) PhysX collision probe, emit traces + MP4s + outcomes. Instrumented with
    flushed progress + a per-scenario wall-clock cap so it cannot hang silently."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("booting Isaac (headless RTX) ...")
    sim = _boot_sim(headless=True)
    _log("Isaac booted; enabling Replicator")
    rep = _enable_and_import_replicator()  # after boot: omni.* now importable + extension enabled
    _log("Replicator ready")
    if disable_physx:
        # NOTE: confirmed on GPU to break the RTX renderer (hangs at render-product creation) —
        # kept only for experiments. Keep PhysX on and use settle_seconds instead.
        _disable_physics_cooking()
        _log("PhysX cooking disabled (WARNING: breaks the renderer on this image)")
    blockers: list[str] = []
    outcomes: list[dict] = []
    physics_contact_reports: list[dict[str, Any]] = []
    result = None
    if dynamic_standing_contact_steps > 0:
        articulated = True
        physics_articulation_drive = True
    try:
        _log(f"opening kitchen USD: {kitchen_usd}")
        stage = _open_stage(_resolve_asset_uri(kitchen_usd))
        if cheap_collision:
            nc = _force_cheap_collision(stage, approximation=collision_approximation)
            _log(f"forced {collision_approximation} collision on {nc} mesh-collision prims")
        _log("kitchen stage open; binding G1 articulation")
        binding = _bind_g1(stage, _resolve_asset_uri(g1_usd))
        _log(f"G1 binding: articulation={binding['controllable_articulation_detected']} "
             f"collision={binding['collision_enabled_verified']}")
        (out_dir / "g1_binding.json").write_text(json.dumps(binding, indent=2))
        if not binding["controllable_articulation_detected"]:
            blockers.append("official_isaac_unitree_g1_articulation_api_unverified")
        if focus_radius > 0:
            route_pts = [p for sc in scenarios for p in sc.get("route_points", [])]
            pr = _prune_to_focus(stage, route_pts, focus_radius, keep_substrings)
            _log(f"focus prune (r={focus_radius}m): kept {pr['kept']} objects, deactivated {pr['pruned']}")
            (out_dir / "focus_prune.json").write_text(json.dumps(pr, indent=2))
        rest_offsets = None
        art_ctx = None
        if articulated:
            rest_offsets = _g1_link_rest_offsets(stage, binding["prim_path"])
            # The physics articulation drive (SimulationContext + SingleArticulation tensor view) is
            # OPT-IN and default-OFF. The crash pattern to avoid is mutating the G1 USD root xform
            # after the tensor view exists. In the physics path, all root seeds go through the
            # articulation API; the pure-USD _place_root fallback is used only when art_ctx is None.
            if physics_articulation_drive:
                try:
                    gravity_z = -9.81 if dynamic_standing_contact_steps > 0 else 0.0
                    art_ctx = _setup_articulated_g1(binding["prim_path"], gravity_z=gravity_z)
                    _log(
                        "G1 articulation drive ready: "
                        f"{art_ctx['dof_count']} joints, {len(art_ctx['link_names'])} links, "
                        f"gravity_z={gravity_z}"
                    )
                except Exception as exc:  # noqa: BLE001
                    blockers.append("official_isaac_unitree_g1_joint_drive_unavailable")
                    _log(f"G1 articulation drive unavailable ({exc!r}); using USD skeleton fallback")
                if art_ctx is not None and not art_ctx.get("link_names"):
                    _log("G1 articulation body/link names unavailable; using USD skeleton fallback for landmarks")
            _log(f"G1 skeleton (USD rest offsets): {len(rest_offsets)} link landmarks")
        from pxr import UsdGeom, UsdLux  # type: ignore
        UsdGeom.Scope.Define(stage, "/World")
        over_cam = "/World/Cameras/overview"
        pov_cam = "/World/Cameras/robot_pov"
        verify_cam_path = "/World/Cameras/verify"
        UsdGeom.Camera.Define(stage, over_cam)
        UsdGeom.Camera.Define(stage, pov_cam)
        if verify_cam:
            UsdGeom.Camera.Define(stage, verify_cam_path)
        key = UsdLux.DistantLight.Define(stage, "/World/Key")
        try:
            key.CreateIntensityAttr(3000.0)  # lift the global key so the workspace is not crushed dark
        except Exception:  # noqa: BLE001
            pass
        # POV camera: widen from USD's ~17deg telephoto default to the projection FOV so the frame
        # shows the lit workspace (not a zoomed crop of the dark basin) AND the rendered view matches
        # the skeleton projection. Overview gets a wide FOV so it frames the whole scene.
        _set_camera_fov(stage, pov_cam, camera_vfov_deg, width, height)
        _set_camera_fov(stage, over_cam, 60.0, width, height)
        if verify_cam:
            _set_camera_fov(stage, verify_cam_path, 55.0, width, height)
        if manipulation_cam and fill_light_intensity > 0 and manipulation_look_at is not None:
            _add_workspace_fill_light(stage, manipulation_look_at, intensity=fill_light_intensity)
            _log(f"workspace fill light @ {tuple(round(float(c),2) for c in manipulation_look_at)} "
                 f"intensity={fill_light_intensity}")
        if neutral_environment:
            try:
                n_dome = _neutralize_environment(stage)
                _log(f"neutralized {n_dome} outdoor-HDRI dome light(s) -> enclosed neutral environment")
            except Exception as exc:  # noqa: BLE001
                _log(f"environment neutralize skipped ({exc!r})")
        _log(f"creating render products ({width}x{height})")
        over_annot = _make_render_product(over_cam, width, height)
        pov_annot = _make_render_product(pov_cam, width, height)
        verify_annot = _make_render_product(verify_cam_path, width, height) if verify_cam else None
        center, radius = scene_framing(scenarios)
        _place_camera(stage, over_cam,
                      (center[0] + radius * 1.4, center[1] - radius * 1.4, center[2] + radius * 1.1),
                      center)
        _log("render products + overview camera ready")
        if settle_seconds > 0:
            # Let PhysX finish async collision-cooking BEFORE we render — rendering *during*
            # cooking is what hangs frame 2+. A pure wait lets the background cook threads drain.
            _log(f"settling {settle_seconds}s for PhysX cooking to drain before rendering")
            t_settle = time.time()
            while time.time() - t_settle < settle_seconds:
                time.sleep(15)
                _log(f"  settle {int(time.time() - t_settle)}/{settle_seconds}s")
            _log("settle complete; starting render")
        if no_collision_probe:
            _log("collision probe DISABLED (policy goes direct every step)")
            def probe(pose, yaw):  # noqa: ANN001
                return 0
        else:
            probe = _overlap_probe(binding["prim_path"])
        for sc in scenarios:
            sid = sc["scenario_id"]
            sdir = out_dir / sid
            (sdir / "frames").mkdir(parents=True, exist_ok=True)
            pol = policy_mod.make_policy(policy_id)
            pol.reset(sc)
            t_sc = time.time()
            _log(f"scenario {sid}: warmup {warmup_frames} render frames (capped {per_scenario_seconds}s)")
            for wi in range(warmup_frames):
                if time.time() - t_sc > per_scenario_seconds:
                    _log(f"warmup hit time cap at frame {wi}")
                    break
                ts = time.time()
                rep.orchestrator.step()
                _log(f"warmup frame {wi} render took {time.time() - ts:.1f}s")
            actions: list[dict] = []
            skel_rows: list[dict] = []
            trace = (sdir / "trace.jsonl").open("w")
            rejected_total = response_total = 0
            cap = 0
            truncated = False
            _log(f"scenario {sid}: stepping {steps}")
            for step in range(steps):
                if time.time() - t_sc > per_scenario_seconds:
                    _log(f"scenario {sid}: per-scenario cap {per_scenario_seconds}s hit at step {step}; truncating")
                    truncated = True
                    break
                ctx = policy_mod.StepContext(step=step, num_steps=steps, probe_collision=probe)
                decision = pol.step(ctx)
                rejected_total += decision.rejected_collision_probe_count
                if decision.policy_action != "accepted_direct_collision_checked_motion":
                    response_total += 1
                route_distance_m = policy_mod.route_distance(sc["route_points"])
                alpha = 0.0 if steps <= 1 else step / float(steps - 1)
                phase = policy_mod.gait_phase(alpha, route_distance_m)
                moving = route_distance_m > 0.05 and step < max(1, steps - 1)
                if art_ctx is not None:
                    if dynamic_standing_contact_steps > 0:
                        report = _settle_dynamic_standing_contacts(
                            stage=stage,
                            art_ctx=art_ctx,
                            robot_prim_path=binding["prim_path"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            settle_steps=dynamic_standing_contact_steps,
                            scenario_id=sid,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                        )
                        report["step"] = step
                        physics_contact_reports.append(report)
                        if report["status"] != "completed":
                            blockers.append("physics_articulation_standing_contact_settle_failed")
                            _log(f"dynamic standing/contact settle failed at step {step}: {report['errors']}")
                    else:
                        _drive_g1_walk(
                            art_ctx["art"],
                            art_ctx["dof_index"],
                            art_ctx["default"],
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                            phase=phase,
                            moving=moving,
                            manipulation_ready=bool(manipulation_reach),
                            manipulation_reach_arm=manipulation_reach_arm,
                        )
                else:
                    _place_root(stage, binding["prim_path"], decision.root_pose, decision.yaw)
                    # Show the arm reaching in the RENDERED frame (pure USD, no physics tensor -> no
                    # crash). The shoulder rotates so the upper arm points at the workspace target.
                    if (kinematic_arm_pose and manipulation_reach
                            and manipulation_look_at is not None):
                        arm_frac = 1.0 if manipulation_cam else alpha
                        try:
                            _pose_arm_kinematic_usd(stage, binding["prim_path"], manipulation_look_at,
                                                    arm=manipulation_reach_arm, reach_frac=arm_frac)
                        except Exception as exc:  # noqa: BLE001 - pose is best-effort, never blocks frames
                            if step == 0:
                                _log(f"kinematic arm pose skipped ({exc!r})")
                rec = policy_mod.action_record(
                    decision=decision, step=step, sim_time_s=step / float(fps), target=sc["target"],
                    scenario_eval_run_id=sc.get("scenario_eval_run_id"))
                actions.append(rec)
                trace.write(json.dumps(rec) + "\n")
                if step % max(1, capture_every) == 0:
                    eye, tgt = (manipulation_cam_pose(decision.root_pose, decision.yaw,
                                                      look_at=manipulation_look_at)
                                if manipulation_cam
                                else follow_cam_pose(decision.root_pose, decision.yaw))
                    _place_camera(stage, pov_cam, eye, tgt)  # POV camera (manipulation egocentric or follow)
                    if verify_annot is not None:
                        v_eye, v_tgt = verify_cam_pose(decision.root_pose, decision.yaw)
                        _place_camera(stage, verify_cam_path, v_eye, v_tgt)  # 3rd-person: SHOW the robot
                    if articulated and (art_ctx is not None or rest_offsets is not None):
                        skel = skeleton_world_for_frame(
                            art_ctx=art_ctx,
                            rest_offsets=rest_offsets,
                            root_pose=decision.root_pose,
                            yaw=decision.yaw,
                        )
                        if manipulation_reach and manipulation_look_at is not None:
                            # For manipulation POVs the first frame is already "task started": arms
                            # visible in the workspace. Navigation/follow shots can still ramp.
                            reach_frac = 1.0 if manipulation_cam else alpha
                            skel = compute_arm_reach_skeleton(skel, manipulation_look_at, reach_frac,
                                                              arm=manipulation_reach_arm)
                        lms = _project_skeleton(skel, eye=eye, target=tgt, up=(0.0, 0.0, 1.0),
                                                vfov_deg=camera_vfov_deg, width=width, height=height)
                        if cap == 0:
                            _log(f"step {step}: skeleton {len(skel)} links -> {len(lms)} landmarks in POV frame")
                        skel_rows.append({
                            "episode_id": sid,
                            "scenario_eval_run_id": sc.get("scenario_eval_run_id") or sid,
                            "step": step, "sim_time_s": round(step / float(fps), 6),
                            "camera": "robot_pov", "landmarks": lms,  # OSCAR reads row["landmarks"]
                            "projected_landmark_count": len(lms)})
                    ts = time.time()
                    # Accumulate N RTX subframes on the static (robot placed) frame to drain the
                    # RayTracedLighting denoiser's grain — a single step leaves heavy noise that an
                    # OSCAR start frame should not inherit.
                    for _ in range(max(1, render_subframes)):
                        rep.orchestrator.step()
                    rdt = time.time() - ts
                    over_ok = _save_rgb(over_annot, sdir / "frames" / f"overview_{cap:04d}.png")
                    _save_rgb(pov_annot, sdir / "frames" / f"robot_pov_{cap:04d}.png")
                    if verify_annot is not None:
                        _save_rgb(verify_annot, sdir / "frames" / f"verify_{cap:04d}.png")
                    if cap == 0 or rdt > 5:
                        _log(f"scenario {sid}: frame {cap} captured (render {rdt:.1f}s, overview_ok={over_ok})")
                    cap += 1
            trace.close()
            if articulated and skel_rows:
                with (sdir / "g1_projected_skeleton_trace.jsonl").open("w") as sf:
                    for r in skel_rows:
                        sf.write(json.dumps(r) + "\n")
                total_lm = sum(r["projected_landmark_count"] for r in skel_rows)
                _log(f"scenario {sid}: skeleton trace {len(skel_rows)} frames, {total_lm} total landmarks")
            scenario_contact_reports = [
                r for r in physics_contact_reports if r.get("scenario_id") == sid
            ]
            if scenario_contact_reports:
                (sdir / "physics_articulation_standing_contact_reports.json").write_text(
                    json.dumps(scenario_contact_reports, indent=2)
                )
            _log(f"scenario {sid}: {cap} frames captured, truncated={truncated}; assembling MP4 + outcome")
            summary = assemble_collision_summary(actions=actions, rejected_probe_total=rejected_total,
                                                 response_event_total=response_total)
            outcome = policy_mod.compute_task_outcome(
                actions=actions, start=sc["start"], target=sc["target"],
                route_distance_m=policy_mod.route_distance(sc["route_points"]),
                collision_summary=summary, bounded_steps=len(actions), model_timestep_s=1.0 / float(fps))
            outcome["frames_captured"] = cap
            outcome["truncated"] = truncated
            outcomes.append(outcome)  # record BEFORE MP4 — MP4 is optional, frames already uploaded
            for name in ("overview", "robot_pov"):
                glob = str(sdir / "frames" / f"{name}_*.png")
                try:
                    subprocess.call(mp4_command(glob, fps, str(sdir / f"{name}.mp4")))
                except Exception as e:  # noqa: BLE001
                    _log(f"mp4 assembly for {name} failed ({e!r}); frames preserved for local assembly")
            _log(f"scenario {sid}: done")
    finally:
        try:
            # SimulationApp.close() can terminate or stall the worker process on some
            # remote runtimes, so persist the collector-visible result before closing Isaac.
            result = build_result(scenarios=scenarios, outcomes=outcomes, policy_id=policy_id,
                                  kitchen_usd=kitchen_usd, g1_usd=g1_usd, blockers=blockers,
                                  physics_articulation_contact_reports=physics_contact_reports)
            (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(result, indent=2))
        finally:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
    assert result is not None
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Isaac G1 kitchen parity eval (GPU)")
    ap.add_argument("--request", help="execution request JSON (scenarios + asset hints)")
    ap.add_argument("--kitchen-usd", help="path/URI to Collected_KitchenRoom/KitchenRoom.usd")
    ap.add_argument("--g1-usd", help="path/URI to the official Isaac G1 USD")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy", default="blueprint_default_walk_to_target_smoke_policy")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--warmup-frames", type=int, default=6)
    ap.add_argument("--capture-every", type=int, default=1)
    ap.add_argument("--no-collision-probe", action="store_true",
                    help="skip the PhysX overlap probe (policy goes direct) — decouples render from physics")
    ap.add_argument("--per-scenario-seconds", type=int, default=480,
                    help="wall-clock cap per scenario so a slow/hung render cannot run forever")
    ap.add_argument("--focus-radius", type=float, default=0.0,
                    help="task-aware scene subset: keep only objects within N m of the route (0=full scene)")
    ap.add_argument("--keep-objects", default="room,floor,wall,ground,ceiling,light",
                    help="comma substrings of object names to always keep (structural shell)")
    ap.add_argument("--settle-seconds", type=int, default=0,
                    help="wait N s after scene load (PhysX on) for cooking to drain before rendering")
    ap.add_argument("--disable-physx", action="store_true",
                    help="(experiment only) disable physx cooking — known to break the renderer")
    ap.add_argument("--cheap-collision", action="store_true",
                    help="force bounding-box collision on all meshes (fast cooking; keeps full scene)")
    ap.add_argument("--articulated", action="store_true",
                    help="drive the G1 joints with the walk gait + emit g1_projected_skeleton_trace.jsonl (for OSCAR)")
    ap.add_argument("--camera-vfov", type=float, default=50.0, help="POV camera vertical FOV (deg) for skeleton projection")
    ap.add_argument("--manipulation-cam", action="store_true",
                    help="egocentric manipulation POV (head looking down-forward at the sink/hands) "
                         "instead of the behind-and-above follow cam — for WAM-ing the task, not navigation")
    ap.add_argument("--manipulation-look-at", default=None,
                    help="fixed world 'x,y,z' the manipulation cam aims at (e.g. the faucet) — pins the "
                         "framing to the known workspace instead of the policy's noisy final yaw")
    ap.add_argument("--render-subframes", type=int, default=1,
                    help="RTX orchestrator steps accumulated per captured frame to denoise grain (e.g. 16)")
    ap.add_argument("--manipulation-reach", action="store_true",
                    help="pose the visible G1 arms into the workspace for manipulation POV review; "
                         "this is posed simulator media, not manipulation-success proof")
    ap.add_argument("--manipulation-reach-arm", default="both", choices=["right", "left", "both"],
                    help="which arm is posed for the task")
    ap.add_argument("--fill-light-intensity", type=float, default=0.0,
                    help="add a sphere fill light over the manipulation workspace (the faucet) at this "
                         "intensity to lift the dark sink basin; 0 disables")
    ap.add_argument("--physics-articulation-drive", action="store_true",
                    help="(opt-in, default off) drive the G1 via the physics articulation tensor view. "
                         "All root seeds stay on the articulation API; the pure-USD root xform fallback "
                         "is used only when this is off.")
    ap.add_argument("--dynamic-standing-contact-steps", type=int, default=0,
                    help="opt-in PhysX standing/contact settle steps per sampled placement. This "
                         "forces --articulated and --physics-articulation-drive, enables gravity, "
                         "and records physics_articulation_standing_contact_reports.json. It is "
                         "standing/contact evidence, not full dynamic walking.")
    ap.add_argument("--neutral-environment", action="store_true",
                    help="replace the kitchen asset's outdoor-HDRI dome light with a neutral bright "
                         "environment (no cityscape through the windows + lifts shadowed surfaces)")
    ap.add_argument("--kinematic-arm-pose", action="store_true",
                    help="pose the RENDERED arm reaching the workspace target via pure-USD shoulder "
                         "rotation (no physics tensor -> crash-safe); needs --manipulation-reach")
    ap.add_argument("--collision-approximation", default="boundingCube",
                    choices=["boundingCube", "convexHull", "convexDecomposition"],
                    help="mesh collision shape: boundingCube (fast, coarse) vs convexHull (shape-"
                         "accurate enough to stand centered + close at the sink, still fast)")
    ap.add_argument("--verify-cam", action="store_true",
                    help="render a 3rd-person verify_*.png that frames the whole robot at the workspace "
                         "(proves where it stands vs the egocentric POV)")
    args = ap.parse_args(argv)

    manip_look_at = None
    if args.manipulation_look_at:
        parts = [float(v) for v in str(args.manipulation_look_at).replace(" ", "").split(",") if v]
        if len(parts) == 3:
            manip_look_at = (parts[0], parts[1], parts[2])

    request = load_request(args.request) if args.request else {}
    scenarios = parse_scenarios(request)
    kitchen_usd = args.kitchen_usd or request.get("kitchen_usd") or request.get("scene_usd")
    g1_usd = args.g1_usd or request.get("g1_usd")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")

    if not scenarios or not kitchen_usd or not g1_usd:
        res = {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked",
               "blockers": ["missing_scenarios_or_kitchen_usd_or_g1_usd"],
               "have_scenarios": bool(scenarios), "have_kitchen_usd": bool(kitchen_usd),
               "have_g1_usd": bool(g1_usd)}
        (out_dir / "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(res, indent=2))
        upload_zip(out_dir, put_url)
        print(json.dumps(res))
        return 1
    result = run_scenarios(
        kitchen_usd=kitchen_usd, g1_usd=g1_usd, scenarios=scenarios, out_dir=out_dir,
        policy_id=args.policy, steps=args.steps, width=args.width, height=args.height,
        fps=args.fps, warmup_frames=args.warmup_frames, capture_every=args.capture_every,
        no_collision_probe=args.no_collision_probe, per_scenario_seconds=args.per_scenario_seconds,
        focus_radius=args.focus_radius,
        keep_substrings=tuple(s for s in args.keep_objects.split(",") if s.strip()),
        disable_physx=args.disable_physx, settle_seconds=args.settle_seconds,
        cheap_collision=args.cheap_collision, articulated=args.articulated,
        camera_vfov_deg=args.camera_vfov, manipulation_cam=args.manipulation_cam,
        manipulation_look_at=manip_look_at, render_subframes=args.render_subframes,
        manipulation_reach=args.manipulation_reach, manipulation_reach_arm=args.manipulation_reach_arm,
        fill_light_intensity=args.fill_light_intensity,
        physics_articulation_drive=args.physics_articulation_drive,
        dynamic_standing_contact_steps=args.dynamic_standing_contact_steps,
        neutral_environment=args.neutral_environment,
        kinematic_arm_pose=args.kinematic_arm_pose,
        collision_approximation=args.collision_approximation, verify_cam=args.verify_cam)
    upload_zip(out_dir, put_url)
    print(json.dumps({"status": result["status"], "passed": result["scenarios_passed"],
                      "executed": result["scenarios_executed"]}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
