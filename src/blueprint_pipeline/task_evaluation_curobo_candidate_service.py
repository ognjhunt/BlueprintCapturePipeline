"""Executable cuRobo v0.8.0 service for sequential task-motion candidates.

This module is installed into the native worker source packet.  It imports
cuRobo lazily so the control plane can build and validate requests without a
GPU.  On the worker it plans every normalized entry/approach/contact/release/
retreat target sequentially, carrying the previous joint result forward.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_collision_aware_candidate_generation import (
    REQUEST_SCHEMA_VERSION,
    REQUIRED_STAGE_KINDS,
    RESULT_SCHEMA_VERSION,
    RUNTIME_PROBE_SCHEMA_VERSION,
)
from .task_evaluation_curobo_candidate_generator import CUROBO_BACKEND_IDENTITY


class CuroboCandidateServiceError(RuntimeError):
    """The pinned runtime or sealed motion-generation input was invalid."""


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")


def _runtime_probe() -> dict[str, Any]:
    try:
        import torch
        import curobo

        version = importlib.metadata.version("nvidia_curobo")
        imported_version = str(curobo.__version__).lstrip("v")
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        raise CuroboCandidateServiceError("curobo_runtime_unavailable") from exc
    source_revision = os.environ.get("BLUEPRINT_CUROBO_SOURCE_REVISION", "")
    if (
        version != CUROBO_BACKEND_IDENTITY["package_version"]
        or imported_version != CUROBO_BACKEND_IDENTITY["package_version"]
        or source_revision != CUROBO_BACKEND_IDENTITY["source_revision"]
        or not cuda_available
        or cuda_count < 1
    ):
        raise CuroboCandidateServiceError("curobo_runtime_identity_invalid")
    result = {
        "schema_version": RUNTIME_PROBE_SCHEMA_VERSION,
        "runtime_ready": True,
        "backend_identity": dict(CUROBO_BACKEND_IDENTITY),
        "cuda_available": True,
        "cuda_device_count": cuda_count,
        "probe_digest": "",
    }
    result["probe_digest"] = canonical_digest(result, digest_field="probe_digest")
    return result


def _read_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CuroboCandidateServiceError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise CuroboCandidateServiceError(blocker)
    return dict(value)


def _referenced_json(reference: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    import hashlib

    path = Path(str(reference.get("path") or ""))
    try:
        resolved = path.resolve(strict=True)
        data = resolved.read_bytes()
    except OSError as exc:
        raise CuroboCandidateServiceError(f"curobo_{role}_unavailable") from exc
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    if (
        resolved.is_symlink()
        or len(data) != reference.get("size_bytes")
        or digest != reference.get("digest")
        or reference.get("role") != role
    ):
        raise CuroboCandidateServiceError(f"curobo_{role}_invalid")
    for attachment in reference.get("attachments") or []:
        attachment_path = Path(str(attachment.get("path") or ""))
        try:
            attachment_data = attachment_path.read_bytes()
        except OSError as exc:
            raise CuroboCandidateServiceError(
                f"curobo_{role}_attachment_invalid"
            ) from exc
        if (
            attachment_path.is_symlink()
            or len(attachment_data) != attachment.get("size_bytes")
            or "sha256:" + hashlib.sha256(attachment_data).hexdigest()
            != attachment.get("digest")
        ):
            raise CuroboCandidateServiceError(
                f"curobo_{role}_attachment_invalid"
            )
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CuroboCandidateServiceError(f"curobo_{role}_invalid") from exc
    if not isinstance(value, Mapping):
        raise CuroboCandidateServiceError(f"curobo_{role}_invalid")
    return dict(value)


def _quat_conjugate_xyzw(quaternion: Sequence[float]) -> list[float]:
    x, y, z, w = (float(value) for value in quaternion)
    return [-x, -y, -z, w]


def _quat_multiply_xyzw(first: Sequence[float], second: Sequence[float]) -> list[float]:
    ax, ay, az, aw = (float(value) for value in first)
    bx, by, bz, bw = (float(value) for value in second)
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    rotated = _quat_multiply_xyzw(
        _quat_multiply_xyzw(quaternion, [*map(float, vector), 0.0]),
        _quat_conjugate_xyzw(quaternion),
    )
    return rotated[:3]


def _world_pose_to_robot_wxyz(
    *, base_pose: Mapping[str, Any], goal: Mapping[str, Any]
) -> list[float]:
    base_position = [float(value) for value in base_pose["position_world_m"]]
    base_quaternion = [float(value) for value in base_pose["orientation_xyzw"]]
    inverse = _quat_conjugate_xyzw(base_quaternion)
    delta = [
        float(goal["position_world_m"][index]) - base_position[index]
        for index in range(3)
    ]
    position = _rotate_xyzw(inverse, delta)
    orientation_xyzw = _quat_multiply_xyzw(
        inverse, [float(value) for value in goal["orientation_world_xyzw"]]
    )
    x, y, z, w = orientation_xyzw
    return [*position, w, x, y, z]


def _sample_joint_waypoints(
    *, positions: Any, joint_names: Sequence[str], maximum: int
) -> list[dict[str, Any]]:
    rows = positions.detach().cpu().tolist()
    if rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        rows = rows[0]
    if not isinstance(rows, list) or not rows:
        raise CuroboCandidateServiceError("curobo_trajectory_missing")
    indices = list(range(len(rows)))
    if len(indices) > maximum:
        indices = sorted(
            {round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)}
        )
    return [
        {
            "waypoint_id": f"solver-{index:04d}",
            "robot_joint_positions_rad": {
                str(name): float(value)
                for name, value in zip(joint_names, rows[index], strict=True)
            },
        }
        for index in indices
    ]


def _generate(request: Mapping[str, Any]) -> dict[str, Any]:
    _runtime_probe()
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("backend_identity") != CUROBO_BACKEND_IDENTITY
        or request.get("required_stage_kinds") != list(REQUIRED_STAGE_KINDS)
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise CuroboCandidateServiceError("curobo_request_invalid")

    robot = _referenced_json(request["robot_configuration"], role="robot_configuration")
    world = _referenced_json(request["world_configuration"], role="world_configuration")
    task = _referenced_json(request["task_trajectory"], role="task_trajectory")
    analytic = _referenced_json(
        request["analytic_candidate_inventory"], role="analytic_candidate_inventory"
    )
    if (
        robot.get("schema_version") != "task_evaluation_curobo_robot_configuration.v1"
        or world.get("schema_version") != "task_evaluation_curobo_world_configuration.v1"
        or task.get("schema_version")
        != "task_evaluation_curobo_normalized_task_trajectory.v1"
        or analytic.get("schema_version")
        != "task_evaluation_curobo_analytic_candidate_inventory.v1"
    ):
        raise CuroboCandidateServiceError("curobo_input_schema_invalid")

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState
    import torch
    import time

    planner = dict(robot.get("planner_configuration") or {})
    permitted_planner_keys = {
        "num_ik_seeds",
        "num_trajopt_seeds",
        "position_tolerance",
        "orientation_tolerance",
        "use_cuda_graph",
        "optimizer_collision_activation_distance",
        "store_debug",
    }
    if any(key not in permitted_planner_keys for key in planner):
        raise CuroboCandidateServiceError("curobo_planner_configuration_invalid")
    joint_names = [str(value) for value in robot.get("joint_names") or []]
    if not joint_names:
        raise CuroboCandidateServiceError("curobo_robot_joint_names_missing")
    candidate_phases = task.get("candidate_phases")
    if not isinstance(candidate_phases, Mapping):
        raise CuroboCandidateServiceError("curobo_task_stages_invalid")
    phases = None
    # Per-base entry variants are part of the immutable candidate universe;
    # select them only after the analytic candidate id is known below.
    candidates = analytic.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CuroboCandidateServiceError("curobo_analytic_candidates_missing")

    solutions: list[dict[str, Any]] = []
    for rank, seed in enumerate(candidates):
        if len(solutions) >= int(request["maximum_candidates"]):
            break
        seed_id = str(seed.get("candidate_id") or "")
        phases = candidate_phases.get(seed_id)
        if (
            not isinstance(phases, list)
            or [row.get("stage_kind") for row in phases]
            != list(REQUIRED_STAGE_KINDS)
        ):
            raise CuroboCandidateServiceError("curobo_task_stages_invalid")
        world_models = world.get("candidate_world_models_robot_frame")
        if not isinstance(world_models, Mapping) or seed_id not in world_models:
            raise CuroboCandidateServiceError("curobo_candidate_world_model_missing")
        torch.manual_seed(int(seed.get("solver_seed", rank)))
        motion_config = MotionPlannerCfg.create(
            robot=robot["curobo_robot_config"],
            scene_model=world_models[seed_id],
            random_seed=int(seed.get("solver_seed", rank)),
            **planner,
        )
        motion_gen = MotionPlanner(motion_config)
        motion_gen.warmup(
            enable_graph=bool(planner.get("use_cuda_graph", True)),
            num_warmup_iterations=int(robot.get("warmup_iterations", 5)),
        )
        reset_map = dict(seed.get("robot_joint_reset_positions_rad") or {})
        if set(reset_map) != set(joint_names):
            raise CuroboCandidateServiceError("curobo_candidate_reset_invalid")
        current = JointState.from_position(
            torch.tensor(
                [[float(reset_map[name]) for name in joint_names]],
                device="cuda:0",
                dtype=torch.float32,
            ),
            joint_names=joint_names,
        )
        stages: list[dict[str, Any]] = []
        solve_time_s = 0.0
        solution_available = True
        for phase in phases:
            stage_waypoints: list[dict[str, Any]] = []
            for goal in phase.get("waypoints") or []:
                robot_pose = _world_pose_to_robot_wxyz(
                    base_pose=seed["robot_base_pose_world"], goal=goal
                )
                position = torch.tensor(
                    robot_pose[:3], device="cuda:0", dtype=torch.float32
                ).view(1, 1, 1, 1, 3)
                quaternion = torch.tensor(
                    robot_pose[3:], device="cuda:0", dtype=torch.float32
                ).view(1, 1, 1, 1, 4)
                goal_pose = GoalToolPose(
                    tool_frames=motion_gen.tool_frames,
                    position=position,
                    quaternion=quaternion,
                )
                started = time.monotonic()
                result = motion_gen.plan_pose(goal_pose, current)
                solve_time_s += time.monotonic() - started
                if result is None or result.success is None or not bool(result.success.any()):
                    solution_available = False
                    break
                path = result.get_interpolated_plan()
                sampled = _sample_joint_waypoints(
                    positions=path.position,
                    joint_names=joint_names,
                    maximum=int(robot.get("maximum_emitted_waypoints_per_stage", 96)),
                )
                for waypoint in sampled:
                    waypoint["target_position_world_m"] = [
                        float(value) for value in goal["position_world_m"]
                    ]
                    waypoint["target_orientation_world_xyzw"] = [
                        float(value) for value in goal["orientation_world_xyzw"]
                    ]
                stage_waypoints.extend(sampled)
                current = JointState.from_position(
                    path.position.reshape(-1, path.position.shape[-1])[-1].view(1, -1),
                    joint_names=joint_names,
                )
            if not solution_available or not stage_waypoints:
                solution_available = False
                break
            stages.append(
                {
                    "stage_id": str(phase["phase_id"]),
                    "stage_kind": str(phase["stage_kind"]),
                    "waypoints": stage_waypoints,
                }
            )
        if not solution_available:
            continue
        addressed = sorted(
            set(request["addressable_feedback_codes"])
            & set(seed.get("addressed_feedback_codes") or [])
        )
        solution = {
            "solution_id": seed_id,
            "deterministic_rank": int(seed.get("deterministic_rank", rank)),
            "source_analytic_candidate_id": seed_id,
            "robot_base_pose_world": dict(seed["robot_base_pose_world"]),
            "support_surface_id": str(seed["support_surface_id"]),
            "robot_joint_reset_positions_rad": reset_map,
            "joins_authored_phase_id": str(task["joins_authored_phase_id"]),
            "stages": stages,
            "cameras": [dict(row) for row in seed["cameras"]],
            "addressed_feedback_codes": addressed,
            # cuRobo v1 exposes constraint-validity, not a signed clearance
            # margin.  Preserve that claim ceiling and let native readback own
            # measured clearance/contact.
            "minimum_world_clearance_m": 0.0,
            "minimum_self_clearance_m": 0.0,
            "joint_limit_compliance_observed": True,
            "collision_aware_motion_generated": True,
            "solver_timing_seconds": solve_time_s,
            "solution_digest": "",
        }
        solution["solution_digest"] = canonical_digest(
            solution, digest_field="solution_digest"
        )
        solutions.append(solution)
    if not solutions:
        raise CuroboCandidateServiceError("curobo_no_collision_aware_solution")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "backend_identity": dict(CUROBO_BACKEND_IDENTITY),
        "request_digest": request["request_digest"],
        "solutions": solutions,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.probe:
            if args.request_json is not None:
                raise CuroboCandidateServiceError("curobo_probe_request_forbidden")
            value = _runtime_probe()
        else:
            if args.request_json is None:
                raise CuroboCandidateServiceError("curobo_request_missing")
            value = _generate(_read_json(args.request_json, blocker="curobo_request_invalid"))
        _write(args.result_json, value)
    except CuroboCandidateServiceError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as the worker entry point
    raise SystemExit(main())


__all__ = ["CuroboCandidateServiceError", "main"]
