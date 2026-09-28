"""Execute the policy-free native Arena rigid-destination placement probe."""

from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULT_FILENAME = "task_evaluation_rigid_destination_native_observation.v1.json"
FAILURE_SCHEMA_VERSION = "task_evaluation_rigid_destination_native_observation_failure.v1"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(blocker) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise RuntimeError(blocker)
    return dict(value)


def _task_support_pose(reset: Mapping[str, Any]) -> list[float]:
    rows = [
        row
        for row in reset.get("objects") or []
        if isinstance(row, Mapping)
        and row.get("asset_id")
        and row.get("task_subject") is False
        and str(row.get("runtime_name") or "").startswith("task_support")
    ]
    if len(rows) != 1:
        raise RuntimeError("destination_qualification_task_support_readback_invalid")
    pose = rows[0].get("observed_root_pose_world") or {}
    values = [
        *[float(item) for item in pose.get("position_world_m") or []],
        *[float(item) for item in pose.get("orientation_xyzw") or []],
    ]
    if len(values) != 7 or not all(math.isfinite(item) for item in values):
        raise RuntimeError("destination_qualification_task_support_pose_invalid")
    return values


def _rotate_xyzw(vector: Sequence[float], quaternion: Sequence[float]) -> list[float]:
    x, y, z, w = (float(item) for item in quaternion)
    vx, vy, vz = (float(item) for item in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _maximum_support_penetration_m(
    *, pose: Sequence[float], collision_bounds: Mapping[str, Any], support_top_z_m: float
) -> float:
    lower = [float(item) for item in collision_bounds.get("minimum") or []]
    upper = [float(item) for item in collision_bounds.get("maximum") or []]
    if (
        len(lower) != 3
        or len(upper) != 3
        or any(low >= high for low, high in zip(lower, upper, strict=True))
    ):
        raise RuntimeError("destination_qualification_collision_bounds_invalid")
    points = [
        _rotate_xyzw([x, y, z], pose[3:])
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]
    minimum_z = min(point[2] + float(pose[2]) for point in points)
    return max(0.0, float(support_top_z_m) - minimum_z)


def _camera_observations(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    rows: list[dict[str, Any]] = []
    for camera in snapshot.get("cameras") or []:
        role = str(camera.get("role") or "")
        calibration = {
            "intrinsic_matrix": camera.get("intrinsic_matrix"),
            "position_world_m": camera.get("position_world_m"),
            "quaternion_world_opengl_xyzw": camera.get(
                "quaternion_world_opengl_xyzw"
            ),
            "native_sensor_timestamp": camera.get("native_sensor_timestamp"),
        }
        render_receipt = {
            "role": role,
            "snapshot_id": snapshot.get("snapshot_id"),
            "rgb_png": camera.get("rgb_png"),
            "camera_calibration": calibration,
            "semantic_label_pixels": camera.get("semantic_label_pixels"),
        }
        rows.append(
            {
                "role": role,
                "task_support_pixel_count": int(
                    (camera.get("semantic_label_pixels") or {}).get(
                        "task_support", 0
                    )
                ),
                "camera_calibration": calibration,
                "render_receipt_digest": canonical_digest(render_receipt),
            }
        )
    return rows


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena_destination_qualification:{phase}:{status}",
        flush=True,
    )


def _capture_destination_measurements(
    *,
    env: Any,
    built: Any,
    request: Mapping[str, Any],
    collision_bounds: Mapping[str, Any],
    support_top_z_m: float,
    torch: Any,
    reset_reader: Any,
    sample_reader: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive repeated native reset/settle/readback without policy actions."""

    seed = int(built.plan["scenario"]["seed"])
    robot = env.unwrapped.scene["robot"]
    settle_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    for index in range(int(request["settle_sample_count"])):
        env.reset(seed=seed)
        reset = reset_reader(built)
        if not reset.get("passed"):
            raise RuntimeError(
                "destination_qualification_repeated_reset_readback_failed"
            )
        reset_pose = _task_support_pose(reset)
        reset_rows.append(
            {"sample_index": index, "destination_pose_world": reset_pose}
        )
        for _ in range(int(request["settle_steps_per_sample"])):
            current = torch.as_tensor(robot.data.joint_pos)[0, :7]
            env.step(
                torch.tensor(
                    [[*[float(value) for value in current], 0.0]],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
            )
        sample = sample_reader.read_task_sample()
        pose = [float(item) for item in sample["destination_pose_world"]]
        row = {
            "sample_index": index,
            "destination_pose_world": pose,
            "maximum_penetration_m": _maximum_support_penetration_m(
                pose=pose,
                collision_bounds=collision_bounds,
                support_top_z_m=support_top_z_m,
            ),
            "support_contact_peak_force_n": float(
                sample["destination_scene_support_contact_peak_force_n"]
            ),
            "forbidden_contact_peak_force_n": float(
                sample["destination_scene_forbidden_contact_peak_force_n"]
            ),
        }
        settle_rows.append(row)
        raw_samples.append({"sample_index": index, "reset": reset, "sample": sample})
    return settle_rows, reset_rows, raw_samples


def main() -> int:
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime / "runtime_output")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / RESULT_FILENAME
    phase = "not_started"
    simulation_app = None
    env = None
    failure: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "blocked",
        "producer": "native_task_arena_destination_qualification",
        "native_isaac_executed": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "policy_loaded": False,
        "phase_reached": phase,
        "blockers": [],
    }
    try:
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _body_pose_world,
            _camera_snapshot,
            _load_and_verify_manifest,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
            _run_wrist_camera_mount_sweep,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeRigidTaskArenaReadback,
            read_native_task_arena_object_reset_state,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )
        from blueprint_pipeline.task_evaluation_rigid_destination_native_observation import (
            materialize_rigid_destination_native_observation,
            validate_rigid_destination_native_probe_request,
        )

        _announce("packet_verification")
        manifest = _load_and_verify_manifest(
            runtime, expected_execution_mode="destination_qualification"
        )
        packet = runtime / "native_task_packet"
        inputs = runtime / "runtime_inputs"
        plan = _load(
            packet / "native_task_arena_scene_plan.v1.json",
            blocker="destination_qualification_scene_plan_invalid",
        )
        packet_request = _load(
            packet / "native_task_arena_packet_request.v1.json",
            blocker="destination_qualification_packet_request_invalid",
        )
        request = validate_rigid_destination_native_probe_request(
            _load(
                inputs / "rigid_destination_native_probe_request.v1.json",
                blocker="destination_qualification_request_invalid",
            )
        )
        static_path = inputs / "destination_static_qualification.v1.json"
        native_path = inputs / "destination_native_import_qualification.v1.json"
        geometry_path = inputs / "destination_geometry.v1.json"
        support_path = inputs / "configured_scene_support_plane.v1.json"
        static = _load(static_path, blocker="destination_qualification_static_invalid")
        native = _load(native_path, blocker="destination_qualification_native_invalid")
        geometry = _load(
            geometry_path, blocker="destination_qualification_geometry_invalid"
        )
        support = _load(
            support_path, blocker="destination_qualification_support_plane_invalid"
        )
        if (
            _sha256(static_path)
            != request["destination_static_qualification_digest"]
            or _sha256(native_path)
            != request["destination_native_import_qualification_digest"]
            or geometry.get("geometry_digest")
            != request["destination_geometry_digest"]
            or _sha256(support_path)
            != request["configured_scene_support_plane_digest"]
            or static.get("replacement_identity")
            != request["destination_identity"]
            or native.get("replacement_identity")
            != request["destination_identity"]
            or geometry.get("destination_identity")
            != request["destination_identity"]
            or geometry.get("pose_world") != request["pose_world"]
            or support.get("schema_version")
            != "task_evaluation_support_plane_input.v1"
        ):
            raise RuntimeError("destination_qualification_input_binding_invalid")
        phase = "packet_verified"
        failure["phase_reached"] = phase
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
            appearance_render_path="particlefield_3d_gaussian_splat",
        )
        failure["native_isaac_executed"] = True
        failure["isaaclab_launch"] = launch
        dependencies = preflight_native_dependency_matrix(
            robot_id=str(plan["robot"]["robot_id"])
        )
        if not dependencies["all_required_available"]:
            raise RuntimeError("destination_qualification_dependencies_failed")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not preconstruction["passed"]:
            raise RuntimeError("destination_qualification_preconstruction_failed")
        built = build_native_task_arena_environment(
            plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        env = built.env
        import omni.usd

        from blueprint_pipeline.native_task_nurec_render_setup import (
            setup_and_warm_native_nurec_renderer,
        )

        warmup = setup_and_warm_native_nurec_renderer(
            simulation_app,
            omni.usd.get_context().get_stage(),
            require_display_referred_particlefield=True,
        )
        if not warmup.get("passed"):
            raise RuntimeError("destination_qualification_nurec_warmup_failed")
        warmup_path = output_root / "destination_renderer_warmup.v1.json"
        warmup_path.write_text(
            json.dumps(warmup, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        phase = "environment_built"
        failure["phase_reached"] = phase
        _announce("simulation_app", "completed")

        import torch

        wrist_selection = _run_wrist_camera_mount_sweep(
            simulation_app=simulation_app,
            env=env,
            built=built,
            packet_request=packet_request,
            plan=plan,
            output_root=output_root,
            torch=torch,
            body_pose_reader=_body_pose_world,
            camera_snapshot=_camera_snapshot,
        )
        if wrist_selection is not None and wrist_selection.get("status") != "selected":
            raise RuntimeError("destination_qualification_wrist_mount_sweep_failed")

        reader = NativeRigidTaskArenaReadback(built)
        collision_bounds = (static.get("observed_structure") or {}).get(
            "collision_bounds_body_frame_m"
        )
        support_top = float(support["top_z_m"])
        settle_rows, reset_rows, raw_samples = _capture_destination_measurements(
            env=env,
            built=built,
            request=request,
            collision_bounds=collision_bounds,
            support_top_z_m=support_top,
            torch=torch,
            reset_reader=read_native_task_arena_object_reset_state,
            sample_reader=reader,
        )
        phase = "native_samples_captured"
        failure["phase_reached"] = phase

        snapshot = _camera_snapshot(
            env=env,
            camera_scene_names=built.camera_scene_names,
            output_root=output_root,
            snapshot_id="destination_qualification",
            framing_expectations=(
                (plan.get("task_object_observability") or {}).get("cameras") or {}
            ),
        )
        camera_rows = _camera_observations(snapshot)
        measurements_path = output_root / "destination_native_measurements.v1.json"
        measurements_path.write_text(
            json.dumps(
                {
                    "schema_version": "task_evaluation_rigid_destination_native_measurements.v1",
                    "settle_samples": settle_rows,
                    "reset_samples": reset_rows,
                    "camera_observations": camera_rows,
                    "camera_snapshot": snapshot,
                    "native_readbacks": raw_samples,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts = [
            {
                "role": "native_measurements",
                "relative_path": measurements_path.relative_to(output_root).as_posix(),
            },
            {
                "role": "camera_snapshot_diagnostics",
                "relative_path": "native_task_camera_snapshot_diagnostics.v1.json",
            },
            {
                "role": "renderer_warmup",
                "relative_path": warmup_path.relative_to(output_root).as_posix(),
            },
        ]
        if wrist_selection is not None:
            artifacts.append(
                {
                    "role": "wrist_camera_mount_selection",
                    "relative_path": "wrist_camera_mount_selection.v1.json",
                }
            )
        for camera in snapshot["cameras"]:
            artifacts.append(
                {
                    "role": f"{camera['role']}_render",
                    "relative_path": camera["rgb_png"]["path"],
                }
            )
        materialize_rigid_destination_native_observation(
            request=request,
            execution_manifest=manifest,
            settle_samples=settle_rows,
            reset_samples=reset_rows,
            camera_observations=camera_rows,
            raw_measurement_artifacts=artifacts,
            artifact_root=output_root,
            output_path=output_path,
        )
        phase = "observation_sealed"
        _announce("observation_sealed", "completed")
    except Exception as exc:  # noqa: BLE001 - retained as typed native evidence
        failure["phase_reached"] = phase
        failure["blockers"].append(
            f"destination_qualification_failed_at_{phase}:{type(exc).__name__}:{exc}"[
                :500
            ]
        )
        failure["traceback"] = traceback.format_exc()[-6000:]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                failure["blockers"].append(
                    f"destination_qualification_env_close_failed:{type(exc).__name__}"
                )
        if not output_path.exists():
            output_path.write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as exc:  # noqa: BLE001
                print(
                    "BLUEPRINT_DESTINATION_QUALIFICATION_CLOSE_WARNING:"
                    f"{type(exc).__name__}",
                    flush=True,
                )
    return 0 if phase == "observation_sealed" else 1


if __name__ == "__main__":  # pragma: no cover - provider entry point
    raise SystemExit(main())


__all__ = [
    "FAILURE_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "_camera_observations",
    "_capture_destination_measurements",
    "_maximum_support_penetration_m",
    "main",
]
