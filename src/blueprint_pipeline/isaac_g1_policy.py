"""Pluggable G1 policy for the Isaac parity eval.

Stage A is a **verbatim port** of the MuJoCo walk-to-target controller
(`mujoco_g1_simulator_command.py`): the route interpolation, the collision-aware candidate
generation, the ``policy_action`` labelling, and the task-outcome contract are identical, so
the Isaac lane runs *the same policy* MuJoCo does. The only sim-specific piece — probing a
candidate pose for scene collision — is injected by the host runner via
``StepContext.probe_collision`` (MuJoCo does ``mj_forward`` + contact count; Isaac does a
PhysX overlap query). The policy itself is pure and sim-agnostic, so it is fully unit-testable
without any simulator.

Stage B slot: :class:`Groot17SonicPolicy` wraps the GR00T N1.7 SONIC VLA
(``LucaFrat/groot-bs16``, embodiment ``UNITREE_G1_SONIC``). It consumes per-step camera+state
observations and returns joint targets; it is GPU-only and fail-closed here (loaded on the
worker, not in this process).

Parity constants and helpers are kept byte-for-byte equivalent to the MuJoCo source; see the
``MUJOCO PARITY`` comments. If the MuJoCo controller changes, update both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "isaac_g1_policy.v1"

# --- MUJOCO PARITY: task thresholds (mujoco_g1_simulator_command.py:30-34) ---
TASK_GOAL_TOLERANCE_M = 0.25
TASK_STUCK_MIN_PROGRESS_RATIO = 0.05
TASK_STUCK_MIN_PROGRESS_M = 0.10
TASK_FALL_ROOT_HEIGHT_M = 0.45
TASK_CLEARANCE_THRESHOLD_M = 0.15

Pose = tuple[float, float, float]


# ----------------------------- pure geometry (ported) -----------------------------

def rounded_pose(pose: Sequence[float]) -> Pose:
    return (round(float(pose[0]), 6), round(float(pose[1]), 6), round(float(pose[2]), 6))


def route_distance(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.sqrt(
            (float(b[0]) - float(a[0])) ** 2
            + (float(b[1]) - float(a[1])) ** 2
            + (float(b[2]) - float(a[2])) ** 2
        )
    return total


def pose_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def interpolate_route(points: Sequence[Sequence[float]], alpha: float) -> tuple[Pose, float, int]:
    """MUJOCO PARITY: mujoco_g1_simulator_command.py:_interpolate_route."""
    if not points:
        return (0.0, 0.0, 0.793), 0.0, 0
    if len(points) == 1:
        p = points[0]
        return (float(p[0]), float(p[1]), float(p[2])), 0.0, 0
    total = route_distance(points)
    if total <= 0:
        p = points[-1]
        return (float(p[0]), float(p[1]), float(p[2])), 0.0, len(points) - 1
    remaining = max(0.0, min(1.0, alpha)) * total
    for segment_index, (a, b) in enumerate(zip(points, points[1:])):
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        seg = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        if seg <= 0:
            continue
        if remaining <= seg:
            la = remaining / seg
            return (ax + (bx - ax) * la, ay + (by - ay) * la, az + (bz - az) * la), \
                math.atan2(by - ay, bx - ax), segment_index
        remaining -= seg
    a, b = points[-2], points[-1]
    return (float(b[0]), float(b[1]), float(b[2])), \
        math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])), len(points) - 2


G1_GAIT_JOINTS = (
    "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
)


def gait_joint_deltas(phase: float, moving: bool) -> dict[str, float]:
    """MUJOCO PARITY: mujoco_g1_simulator_command.py:_apply_preview_gait_pose. Per-joint deltas
    (radians, added to the standing pose) for a procedural walk cycle at ``phase``. Left/right
    are anti-phase. Drives the Isaac G1 articulation joints to make it actually walk."""
    if not moving:
        return {}
    stride = math.sin(phase)
    counter = math.sin(phase + math.pi)
    knee_left = max(0.0, math.sin(phase + 0.45)) * 0.35
    knee_right = max(0.0, math.sin(phase + math.pi + 0.45)) * 0.35
    return {
        "left_hip_pitch_joint": 0.24 * stride, "right_hip_pitch_joint": 0.24 * counter,
        "left_knee_joint": knee_left, "right_knee_joint": knee_right,
        "left_ankle_pitch_joint": -0.10 * stride, "right_ankle_pitch_joint": -0.10 * counter,
        "left_shoulder_pitch_joint": -0.20 * stride, "right_shoulder_pitch_joint": -0.20 * counter,
        "left_elbow_joint": 0.08 * max(0.0, -stride), "right_elbow_joint": 0.08 * max(0.0, -counter),
        "waist_yaw_joint": 0.04 * math.sin(phase * 0.5),
    }


def gait_phase(alpha: float, route_distance_m: float) -> float:
    """MUJOCO PARITY: phase = alpha * max(1, route_distance) * 2*pi (mujoco loop)."""
    return alpha * max(1.0, route_distance_m) * math.pi * 2.0


def candidate_pose_specs(
    *, desired_pose: Sequence[float], previous_pose: Sequence[float] | None, yaw: float,
    previous_yaw: float | None = None,
) -> list[dict[str, Any]]:
    """MUJOCO PARITY: mujoco_g1_simulator_command.py:_candidate_pose_specs.
    direct -> lateral redirects (+/-0.18/0.36/0.6) -> stop (if moving) or spawn_relocation ring."""
    x, y, z = float(desired_pose[0]), float(desired_pose[1]), float(desired_pose[2])
    normal = (-math.sin(yaw), math.cos(yaw))
    specs: list[dict[str, Any]] = [{"candidate_kind": "direct", "pose": (x, y, z), "lateral_offset_m": 0.0}]
    for offset in (0.18, -0.18, 0.36, -0.36, 0.6, -0.6):
        specs.append({"candidate_kind": "redirect",
                      "pose": (x + normal[0] * offset, y + normal[1] * offset, z),
                      "lateral_offset_m": offset})
    if previous_pose is not None:
        specs.append({"candidate_kind": "stop",
                      "pose": (float(previous_pose[0]), float(previous_pose[1]), float(previous_pose[2])),
                      "yaw": previous_yaw if previous_yaw is not None else yaw,
                      "lateral_offset_m": 0.0})
    else:
        for radius in (0.35, 0.7, 1.05, 1.4, 1.8):
            for angle_index in range(8):
                angle = yaw + angle_index * (math.pi / 4.0)
                specs.append({"candidate_kind": "spawn_relocation",
                              "pose": (x + math.cos(angle) * radius, y + math.sin(angle) * radius, z),
                              "lateral_offset_m": None, "relocation_radius_m": radius})
    return specs


# ----------------------------- policy interface -----------------------------

@dataclass
class StepContext:
    """What a policy sees this step. ``probe_collision(pose, yaw) -> scene_contact_count`` is the
    host-sim collision oracle (Stage A). Camera/state slots feed the Stage-B VLA."""
    step: int
    num_steps: int
    probe_collision: Callable[[Pose, float], int] | None = None
    camera_rgb: Any = None
    joint_state: Any = None
    instruction: str = ""


@dataclass
class StepDecision:
    root_pose: Pose
    yaw: float
    desired_root_position: Pose
    route_segment_index: int
    policy_action: str
    collision_probe_candidate_count: int
    rejected_collision_probe_count: int
    rejected_probes: list = field(default_factory=list)
    joint_targets: Any = None  # Stage B (GR00T) — None for the kinematic controller


class G1Policy:
    policy_id = "base"

    def reset(self, scenario: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def step(self, ctx: StepContext) -> StepDecision:
        raise NotImplementedError


# ----------------------------- Stage A: deterministic controller -----------------------------

class DeterministicWalkToTargetPolicy(G1Policy):
    """Verbatim port of the MuJoCo collision-aware walk-to-target preview controller.
    Same ``policy_id`` as MuJoCo so the parity is explicit in the trace."""

    policy_id = "blueprint_default_walk_to_target_smoke_policy"

    def reset(self, scenario: Mapping[str, Any]) -> None:
        self.route_points = [tuple(float(c) for c in p) for p in scenario["route_points"]]
        self.start = tuple(float(c) for c in scenario["start"])
        self.target = tuple(float(c) for c in scenario["target"])
        self.route_distance = route_distance(self.route_points)
        self._accepted_pose: Pose | None = None
        self._accepted_yaw = 0.0

    def step(self, ctx: StepContext) -> StepDecision:
        n = ctx.num_steps
        alpha = 0.0 if n <= 1 else ctx.step / float(n - 1)
        desired_pose, yaw, seg = interpolate_route(self.route_points, alpha)
        specs = candidate_pose_specs(
            desired_pose=desired_pose, previous_pose=self._accepted_pose,
            yaw=yaw if self._accepted_pose is None else self._accepted_yaw,
            previous_yaw=self._accepted_yaw,
        )
        probe = ctx.probe_collision or (lambda pose, yaw: 0)
        rejected: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        probed = 0
        for spec in specs:
            probed += 1
            cnt = int(probe(spec["pose"], float(spec.get("yaw", yaw))))
            if cnt > 0:
                rejected.append({
                    "candidate_kind": spec["candidate_kind"],
                    "pose": rounded_pose(spec["pose"]),
                    "lateral_offset_m": spec.get("lateral_offset_m"),
                    "relocation_radius_m": spec.get("relocation_radius_m"),
                    "scene_collision_contact_count": cnt,
                })
            else:
                selected = spec
                break
        if selected is None:
            selected = specs[-1]  # forced: no collision-free pose found
        kind = selected["candidate_kind"]
        if kind == "direct":
            action = "accepted_direct_collision_checked_motion"
        elif kind in {"redirect", "spawn_relocation"}:
            action = "redirected_by_collision_probe"
        else:
            action = "stopped_by_collision_probe"
        self._accepted_pose = (float(selected["pose"][0]), float(selected["pose"][1]), float(selected["pose"][2]))
        self._accepted_yaw = float(selected.get("yaw", yaw))
        return StepDecision(
            root_pose=self._accepted_pose, yaw=self._accepted_yaw,
            desired_root_position=(float(desired_pose[0]), float(desired_pose[1]), float(desired_pose[2])),
            route_segment_index=seg, policy_action=action,
            collision_probe_candidate_count=probed, rejected_collision_probe_count=len(rejected),
            rejected_probes=rejected,
        )


# ----------------------------- Stage B: GR00T N1.7 SONIC (GPU-only slot) -----------------------------

class Groot17SonicPolicy(G1Policy):
    """Closed-loop GR00T N1.7 SONIC VLA (LucaFrat/groot-bs16). GPU-only; loads the 3B
    checkpoint + the NVIDIA gr00t inference stack on the worker. Fail-closed off-GPU."""

    policy_id = "unitree_groot_n17_sonic_policy"
    DEFAULT_CHECKPOINT = "LucaFrat/groot-bs16"
    EMBODIMENT_TAG = "UNITREE_G1_SONIC"

    def __init__(self, checkpoint: str | None = None) -> None:
        self.checkpoint = checkpoint or self.DEFAULT_CHECKPOINT
        self._engine = None

    def available(self) -> dict:
        try:
            import gr00t  # type: ignore  # noqa: F401
        except Exception:  # noqa: BLE001
            return {"available": False, "reason": "gr00t_inference_stack_unavailable",
                    "checkpoint": self.checkpoint}
        return {"available": True, "reason": None, "checkpoint": self.checkpoint}

    def reset(self, scenario: Mapping[str, Any]) -> None:
        self.instruction = str(scenario.get("instruction") or scenario.get("task_text") or "")
        avail = self.available()
        if not avail.get("available"):
            raise RuntimeError(f"groot_sonic_unavailable:{avail.get('reason')}")
        # Real load happens on the GPU worker (gr00t policy server / inference engine).
        raise NotImplementedError("Groot17SonicPolicy.reset: GPU-worker load not implemented in this process")

    def step(self, ctx: StepContext) -> StepDecision:
        raise NotImplementedError("Groot17SonicPolicy.step runs only on the GPU worker")


# ----------------------------- registry -----------------------------

def make_policy(policy_id: str | None = None, **kwargs) -> G1Policy:
    key = (policy_id or DeterministicWalkToTargetPolicy.policy_id).strip()
    if key in {DeterministicWalkToTargetPolicy.policy_id, "deterministic", "walk_to_target"}:
        return DeterministicWalkToTargetPolicy()
    if key in {Groot17SonicPolicy.policy_id, "groot", "groot_sonic", "groot_n17_sonic"}:
        return Groot17SonicPolicy(checkpoint=kwargs.get("checkpoint"))
    raise ValueError(f"unknown_policy_id:{policy_id!r} (known: "
                     f"{DeterministicWalkToTargetPolicy.policy_id}, {Groot17SonicPolicy.policy_id})")


# ----------------------------- task outcome (ported) -----------------------------

def compute_task_outcome(
    *, actions: Sequence[Mapping[str, Any]], start: Sequence[float], target: Sequence[float],
    route_distance_m: float, collision_summary: Mapping[str, Any], bounded_steps: int,
    model_timestep_s: float,
) -> dict[str, Any]:
    """MUJOCO PARITY: mujoco_g1_simulator_command.py:_attempt_task_outcome (identical contract)."""
    start_pose = rounded_pose(start)
    target_pose = rounded_pose(target)
    root_positions = [tuple(a["root_position"]) for a in actions if a.get("root_position") is not None]
    desired_positions = [tuple(a["desired_root_position"]) for a in actions
                         if a.get("desired_root_position") is not None]
    final_pose = root_positions[-1] if root_positions else start_pose
    direct_distance_m = pose_distance(start_pose, target_pose)
    final_target_error_m = pose_distance(final_pose, target_pose)
    actual_path_distance_m = route_distance(root_positions) if len(root_positions) > 1 else 0.0
    progress_m = max(0.0, direct_distance_m - final_target_error_m)
    progress_ratio = progress_m / direct_distance_m if direct_distance_m > 0 else 1.0
    path_deviations = [pose_distance(rp, dp) for rp, dp in zip(root_positions, desired_positions)]
    max_path_deviation_m = max(path_deviations) if path_deviations else 0.0
    mean_path_deviation_m = sum(path_deviations) / len(path_deviations) if path_deviations else 0.0
    z_values = [p[2] for p in root_positions]
    min_root_height_m = min(z_values) if z_values else start_pose[2]
    goal_reached = final_target_error_m <= TASK_GOAL_TOLERANCE_M
    scene_contact_count = int(collision_summary.get("robot_scene_contact_event_count") or 0)
    rejected_probe_count = int(collision_summary.get("rejected_scene_collision_probe_count") or 0)
    near_miss_event_count = int(collision_summary.get("near_miss_event_count") or rejected_probe_count)
    min_clearance_raw = collision_summary.get("min_clearance_m")
    min_clearance_m = round(float(min_clearance_raw), 6) if isinstance(min_clearance_raw, (int, float)) else None
    clearance_threshold_m = float(collision_summary.get("clearance_threshold_m") or TASK_CLEARANCE_THRESHOLD_M)
    if min_clearance_m is None and not near_miss_event_count and scene_contact_count == 0:
        min_clearance_m = round(clearance_threshold_m, 6)
    clearance_threshold_violation = bool(
        collision_summary.get("clearance_threshold_violation")
        or near_miss_event_count > 0
        or (min_clearance_m is not None and min_clearance_m < clearance_threshold_m)
    )
    response_count = int(collision_summary.get("collision_response_event_count") or 0)
    stopped_steps = sum(1 for a in actions if str(a.get("policy_action")) == "stopped_by_collision_probe")
    redirected_steps = sum(1 for a in actions if str(a.get("policy_action")) == "redirected_by_collision_probe")
    fall_detected = bool(min_root_height_m < TASK_FALL_ROOT_HEIGHT_M)
    timeout = not goal_reached
    stuck_detected = bool(
        not goal_reached and direct_distance_m > TASK_STUCK_MIN_PROGRESS_M
        and (progress_m < TASK_STUCK_MIN_PROGRESS_M or progress_ratio < TASK_STUCK_MIN_PROGRESS_RATIO
             or stopped_steps >= max(1, int(len(actions) * 0.5)))
    )
    endpoint_clean = bool(goal_reached and scene_contact_count == 0)
    spawn_clean = bool(not actions or str(actions[0].get("policy_action")) != "redirected_by_collision_probe")
    policy_instability = bool(len(actions) > 0 and (stopped_steps + redirected_steps) / len(actions) > 0.75)
    failure_mode_ids: list[str] = []
    if scene_contact_count:
        failure_mode_ids.append("failure_scene_collision_contact")
    if fall_detected:
        failure_mode_ids.append("failure_robot_fall_detected")
    if not goal_reached:
        failure_mode_ids.append("failure_target_not_reached")
    if not endpoint_clean:
        failure_mode_ids.append("failure_endpoint_not_clean")
    if stuck_detected:
        failure_mode_ids.append("failure_stuck_or_no_progress")
    if timeout:
        failure_mode_ids.append("failure_timeout")
    if policy_instability:
        failure_mode_ids.append("failure_policy_instability")
    if clearance_threshold_violation:
        failure_mode_ids.append("failure_clearance_near_miss")
    success = not failure_mode_ids
    return {
        "task_success": success,
        "task_status": "passed" if success else "failed_task_criteria",
        "failure_mode_ids": failure_mode_ids,
        "failure_reason": ",".join(failure_mode_ids) if failure_mode_ids else None,
        "goal_reached": goal_reached,
        "endpoint_clean": endpoint_clean,
        "spawn_clean": spawn_clean,
        "timeout": timeout,
        "fall_detected": fall_detected,
        "stuck_detected": stuck_detected,
        "policy_instability_detected": policy_instability,
        "final_pose": [round(float(v), 6) for v in final_pose],
        "final_target_error_m": round(final_target_error_m, 6),
        "goal_tolerance_m": TASK_GOAL_TOLERANCE_M,
        "min_clearance_m": min_clearance_m,
        "clearance_threshold_m": clearance_threshold_m,
        "clearance_threshold_violation": clearance_threshold_violation,
        "direct_start_to_target_distance_m": round(direct_distance_m, 6),
        "planned_route_distance_m": round(float(route_distance_m), 6),
        "actual_path_distance_m": round(actual_path_distance_m, 6),
        "path_efficiency_ratio": round(actual_path_distance_m / route_distance_m, 6) if route_distance_m > 0 else None,
        "progress_to_goal_m": round(progress_m, 6),
        "progress_to_goal_ratio": round(progress_ratio, 6),
        "max_path_deviation_m": round(max_path_deviation_m, 6),
        "mean_path_deviation_m": round(mean_path_deviation_m, 6),
        "min_root_height_m": round(float(min_root_height_m), 6),
        "stopped_step_count": stopped_steps,
        "redirected_step_count": redirected_steps,
        "near_miss_event_count": near_miss_event_count,
        "collision_response_event_count": response_count,
        "robot_scene_contact_event_count": scene_contact_count,
        "simulated_step_count": bounded_steps,
        "cycle_time_seconds": round(bounded_steps * model_timestep_s, 6) if model_timestep_s else None,
        "success_criteria": {
            "goal_reached_within_tolerance": goal_reached,
            "goal_tolerance_m": TASK_GOAL_TOLERANCE_M,
            "no_committed_scene_collision_contacts": scene_contact_count == 0,
            "no_clearance_near_miss": not clearance_threshold_violation,
            "no_fall_detected": not fall_detected,
            "no_stuck_or_no_progress": not stuck_detected,
            "endpoint_clean": endpoint_clean,
        },
        "proof_boundary": (
            "Task outcome is computed from the deterministic collision-aware preview trace "
            "(Isaac PhysX candidate probes). Parity with the MuJoCo preview controller; it is "
            "not a learned-policy ranking and not dynamic locomotion."
        ),
    }


def action_record(*, decision: StepDecision, step: int, sim_time_s: float, target: Sequence[float],
                  committed_scene_contact_count: int = 0, contact_count: int = 0,
                  scenario_eval_run_id: str | None = None) -> dict[str, Any]:
    """Build the per-step action record in the MuJoCo trace schema (consumed by compute_task_outcome)."""
    x, y, z = decision.root_pose
    dp = decision.desired_root_position
    return {
        "step": step,
        "sim_time_s": round(float(sim_time_s), 9),
        "root_position": [round(x, 6), round(y, 6), round(z, 6)],
        "desired_root_position": [round(float(dp[0]), 6), round(float(dp[1]), 6), round(float(dp[2]), 6)],
        "root_yaw_radians": round(decision.yaw, 6),
        "target": [round(float(c), 6) for c in target],
        "route_segment_index": decision.route_segment_index,
        "contact_count": contact_count,
        "scene_collision_contact_count": committed_scene_contact_count,
        "collision_probe_candidate_count": decision.collision_probe_candidate_count,
        "rejected_collision_probe_count": decision.rejected_collision_probe_count,
        "policy_action": decision.policy_action,
        "scenario_eval_run_id": scenario_eval_run_id,
    }
