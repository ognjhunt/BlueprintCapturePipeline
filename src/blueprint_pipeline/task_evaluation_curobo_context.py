"""Materialize sealed cuRobo inputs from one production native packet."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pxr import Usd

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_construction_plan import (
    materialize_native_task_construction_phase_plan,
)
from .scene_placement.robot_profile import get_robot_profile
from .task_evaluation_collision_aware_candidate_generation import (
    CandidateGeneratorContext,
)
from .task_evaluation_robot_placement_geometry import _stage_triangles


class CuroboContextError(ValueError):
    """The production packet cannot produce exact cuRobo inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CuroboContextError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise CuroboContextError(blocker)
    return dict(value)


def _reference(path: Path, *, role: str, attachments: list[dict] | None = None) -> dict:
    result = {
        "role": role,
        "path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "digest": _sha256(path),
    }
    if attachments:
        result["attachments"] = attachments
    return result


def _inverse_pose_wxyz(pose: Mapping[str, Any]) -> list[float]:
    position = [float(value) for value in pose["position_world_m"]]
    x, y, z, w = (float(value) for value in pose["orientation_xyzw"])
    inverse = [-x, -y, -z, w]
    vx, vy, vz = (-position[0], -position[1], -position[2])
    qx, qy, qz, qw = inverse
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    translated = [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]
    return [*translated, w, -x, -y, -z]


def _write_world_obj(scene_path: Path, output_path: Path) -> None:
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise CuroboContextError("curobo_scene_collision_usd_invalid")
    triangles, _paths = _stage_triangles(stage)
    if not len(triangles):
        raise CuroboContextError("curobo_scene_collision_triangles_missing")
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Blueprint exact world-space collision triangles\n")
        index = 1
        for triangle in triangles:
            for vertex in triangle:
                stream.write(
                    "v " + " ".join(format(float(value), ".12g") for value in vertex) + "\n"
                )
            stream.write(f"f {index} {index + 1} {index + 2}\n")
            index += 3


def _candidate_rows(universe: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict]:
    raw = universe.get("candidates") if isinstance(universe, Mapping) else universe
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise CuroboContextError("curobo_candidate_universe_invalid")
    rows = [json.loads(json.dumps(dict(row), allow_nan=False)) for row in raw]
    if any(not row.get("candidate_id") for row in rows):
        raise CuroboContextError("curobo_candidate_universe_invalid")
    return rows


def _world_goal(phase: Mapping[str, Any], *, waypoint_id: str) -> dict[str, Any]:
    return {
        "waypoint_id": waypoint_id,
        "position_world_m": [float(value) for value in phase["position_world_m"]],
        "orientation_world_xyzw": [
            float(value) for value in phase["orientation_world_xyzw"]
        ],
    }


def _five_stages(phases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in phases]
    if not rows:
        raise CuroboContextError("curobo_native_phases_missing")
    first = rows[0]
    contact = [
        row
        for row in rows[1:]
        if not any(
            token in str(row.get("phase_id") or "")
            for token in ("release", "settle", "retreat", "recovery")
        )
    ]
    releases = [row for row in rows if "release" in str(row.get("phase_id") or "")]
    retreats = [
        row
        for row in rows
        if any(
            token in str(row.get("phase_id") or "")
            for token in ("retreat", "recovery")
        )
    ]
    if not contact or not releases or not retreats:
        raise CuroboContextError("curobo_native_phase_mapping_incomplete")
    grouped = (
        ("entry", [first]),
        ("approach", [first]),
        ("contact", contact),
        ("release", releases),
        ("retreat", retreats),
    )
    return [
        {
            "phase_id": f"curobo-{kind}",
            "stage_kind": kind,
            "waypoints": [
                _world_goal(row, waypoint_id=str(row["phase_id"])) for row in selected
            ],
        }
        for kind, selected in grouped
    ]


def materialize_remote_curobo_context(
    *,
    packet_dir: str | Path,
    universe: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    output_root: str | Path,
    commit: str,
    maximum_incremental_cost_usd: float = 0.2,
    maximum_runtime_seconds: float = 300.0,
    warm_session: Mapping[str, Any] | None = None,
) -> tuple[CandidateGeneratorContext, str]:
    """Build all four self-contained planner documents and exact mesh attachment."""

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CuroboContextError("curobo_context_commit_invalid")
    packet = Path(packet_dir).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise CuroboContextError("curobo_context_output_exists")
    destination.mkdir(parents=True)
    scene = _read(
        packet / "native_task_arena_scene_plan.v1.json",
        blocker="curobo_scene_plan_invalid",
    )
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene.get("plan_digest")
        != canonical_digest(scene, digest_field="plan_digest")
    ):
        raise CuroboContextError("curobo_scene_plan_invalid")
    plan = materialize_native_task_construction_phase_plan(scene)
    candidates = _candidate_rows(universe)
    profile = get_robot_profile(str((scene.get("robot") or {}).get("robot_id") or ""))
    if profile.robot_id != "franka_panda":
        raise CuroboContextError("curobo_context_robot_unsupported")

    collision_rows = [
        row
        for row in scene.get("objects") or []
        if isinstance(row, Mapping) and row.get("semantic_role") == "scene_collision"
    ]
    if len(collision_rows) != 1:
        raise CuroboContextError("curobo_scene_collision_binding_invalid")
    collision = collision_rows[0]
    source_collision = Path(str(scene["asset_directory"])) / str(collision["filename"])
    try:
        source_collision = source_collision.resolve(strict=True)
    except OSError as exc:
        raise CuroboContextError("curobo_scene_collision_missing") from exc
    if collision.get("sha256") != _sha256(source_collision):
        raise CuroboContextError("curobo_scene_collision_digest_mismatch")
    mesh_path = destination / "scene-collision-world.obj"
    _write_world_obj(source_collision, mesh_path)

    robot_doc = {
        "schema_version": "task_evaluation_curobo_robot_configuration.v1",
        "curobo_robot_config": "franka.yml",
        "joint_names": list(profile.arm_joint_names),
        "planner_configuration": {
            "num_ik_seeds": 64,
            "num_trajopt_seeds": 8,
            "position_tolerance": 0.005,
            "orientation_tolerance": 0.05,
            "optimizer_collision_activation_distance": 0.02,
            "use_cuda_graph": True,
            "store_debug": False,
        },
        "warmup_iterations": 5,
        "maximum_emitted_waypoints_per_stage": 64,
    }
    world_models = {}
    analytic_rows = []
    candidate_phases = {}
    for rank, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        pose = dict(candidate["robot_base_pose_world"])
        world_models[candidate_id] = {
            "mesh": {
                "configured_site_collision": {
                    "file_path": str(mesh_path),
                    "pose": _inverse_pose_wxyz(pose),
                }
            }
        }
        reset = dict(candidate.get("reset_variant") or {})
        cameras = dict(candidate.get("camera_variant") or {})
        reset_positions = dict(reset.get("robot_joint_reset_positions_rad") or {})
        arm_reset = {name: reset_positions[name] for name in profile.arm_joint_names}
        analytic_rows.append(
            {
                "candidate_id": candidate_id,
                "deterministic_rank": int(candidate.get("deterministic_rank", rank)),
                "solver_seed": rank,
                "robot_base_pose_world": pose,
                "support_surface_id": str(candidate["support_surface_id"]),
                "robot_joint_reset_positions_rad": arm_reset,
                "cameras": list(cameras.get("cameras") or []),
                "addressed_feedback_codes": list(
                    candidate.get("addressed_feedback_codes") or []
                ),
            }
        )
        stages = _five_stages(plan["phases"])
        entry_variant = dict(candidate.get("entry_trajectory_variant") or {})
        entry_rows = entry_variant.get("waypoints")
        if not isinstance(entry_rows, list) or not entry_rows:
            raise CuroboContextError("curobo_candidate_entry_trajectory_invalid")
        stages[0]["waypoints"] = [
            {
                "waypoint_id": str(row.get("waypoint_id") or f"entry-{index:02d}"),
                "position_world_m": [
                    float(value) for value in row["position_world_m"]
                ],
                "orientation_world_xyzw": [
                    float(value) for value in row["orientation_world_xyzw"]
                ],
            }
            for index, row in enumerate(entry_rows)
            if isinstance(row, Mapping)
        ]
        if len(stages[0]["waypoints"]) != len(entry_rows):
            raise CuroboContextError("curobo_candidate_entry_trajectory_invalid")
        candidate_phases[candidate_id] = stages

    world_doc = {
        "schema_version": "task_evaluation_curobo_world_configuration.v1",
        "source_scene_plan_digest": scene["plan_digest"],
        "source_scene_collision_digest": collision["sha256"],
        "candidate_world_models_robot_frame": world_models,
    }
    task_doc = {
        "schema_version": "task_evaluation_curobo_normalized_task_trajectory.v1",
        "source_native_phase_plan_digest": plan["plan_digest"],
        "joins_authored_phase_id": str(plan["phases"][0]["phase_id"]),
        "candidate_phases": candidate_phases,
    }
    analytic_doc = {
        "schema_version": "task_evaluation_curobo_analytic_candidate_inventory.v1",
        "source_inventory_digest": (
            universe.get("inventory_digest") if isinstance(universe, Mapping) else None
        ),
        "candidates": analytic_rows,
    }
    paths = {}
    for role, value in (
        ("robot_configuration", robot_doc),
        ("world_configuration", world_doc),
        ("task_trajectory", task_doc),
        ("analytic_candidate_inventory", analytic_doc),
    ):
        path = destination / f"{role}.json"
        write_json(path, value)
        paths[role] = path
    mesh_reference = _reference(mesh_path, role="world_collision_mesh")
    context = CandidateGeneratorContext(
        run_id=str(
            (universe.get("run_id") if isinstance(universe, Mapping) else None)
            or scene.get("task_id")
            or scene.get("scene_id")
            or ""
        ),
        expected_production_commit=commit,
        robot_configuration=_reference(paths["robot_configuration"], role="robot_configuration"),
        world_configuration=_reference(
            paths["world_configuration"],
            role="world_configuration",
            attachments=[mesh_reference],
        ),
        task_trajectory=_reference(paths["task_trajectory"], role="task_trajectory"),
        analytic_candidate_inventory=_reference(
            paths["analytic_candidate_inventory"],
            role="analytic_candidate_inventory",
        ),
        maximum_incremental_cost_usd=float(maximum_incremental_cost_usd),
        maximum_runtime_seconds=float(maximum_runtime_seconds),
    )
    remote_work_dir = str((warm_session or {}).get("remote_work_dir") or "/workspace")
    if remote_work_dir not in {"/workspace", "/tmp/blueprint_vast_work"}:
        raise CuroboContextError("curobo_remote_work_dir_invalid")
    return context, remote_work_dir + "/adp_arena_provider_bundle/provider_runtime"


__all__ = ["CuroboContextError", "materialize_remote_curobo_context"]
