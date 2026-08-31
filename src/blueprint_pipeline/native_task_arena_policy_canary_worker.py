"""Provider worker for one warm, paired internal-policy canary session."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    PROVIDER_RESULT_FILENAME,
    execute_paired_session,
    validate_runtime_input_manifest,
    validate_session_authority,
)


def _digest(value: Any) -> str:
    return canonical_digest({"value": value})


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"policy_canary_input_not_object:{path.name}")
    return value


def _resolved_scene_plan(base: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(base))
    scenario = deepcopy(dict(cell["resolved_scenario"]))
    scenario["cell_id"] = cell["cell_id"]
    scenario["seed"] = cell["seed"]
    plan["scenario"] = scenario
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_episode_json_artifact(
    output_root: Path, *, episode_id: str, role: str, value: Any
) -> dict[str, Any]:
    path = output_root / "episodes" / f"{episode_id}.{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_media_artifact(
    output_root: Path,
    *,
    artifacts: Any,
    role: str,
    role_match: Callable[[str], bool],
) -> dict[str, Any] | None:
    matches = [
        row
        for row in artifacts or []
        if isinstance(row, Mapping) and role_match(str(row.get("role") or ""))
    ]
    if not matches:
        return None
    row = matches[0]
    path = (output_root / str(row.get("relative_path") or "")).resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        return None
    if path.is_symlink() or not path.is_file():
        return None
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_indexed_telemetry(
    output_root: Path, episodes: list[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        {
            "run_kind": episode.get("run_kind"),
            "candidate_id": episode.get("candidate_id"),
            "cell_id": episode.get("cell_id"),
            "seed": episode.get("seed"),
            "telemetry": episode.get("telemetry"),
        }
        for episode in episodes
    ]
    telemetry_path = output_root / "policy_canary_telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    channels = {
        "observations": sum(bool((row.get("telemetry") or {}).get("channels")) for row in rows),
        "episode_envelopes": len(rows),
    }
    schema = {
        "schema_version": "policy_canary_telemetry_schema.v1",
        "timebase": "unix_ns",
        "channels": {
            "episode_envelopes": "policy_canary_episode_telemetry.v1",
            "observations": "native_policy_observation_manifest_reference.v1",
        },
    }
    schema_path = output_root / "policy_canary_telemetry_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    primary_path = telemetry_path
    primary_format = "typed_jsonl"
    mcap_gap: str | None = None
    try:
        from mcap.writer import Writer

        mcap_path = output_root / "policy_canary_telemetry.mcap"
        with mcap_path.open("wb") as stream:
            writer = Writer(stream)
            writer.start(profile="blueprint-policy-canary", library="blueprint_pipeline")
            schema_id = writer.register_schema(
                name="policy_canary_episode_telemetry.v1",
                encoding="jsonschema",
                data=json.dumps(schema, sort_keys=True).encode("utf-8"),
            )
            channel_id = writer.register_channel(
                topic="/blueprint/policy_canary/episode",
                message_encoding="json",
                schema_id=schema_id,
            )
            for row in rows:
                telemetry = row.get("telemetry") or {}
                timestamp = int(telemetry.get("completed_at_unix_ns") or time.time_ns())
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp,
                    publish_time=timestamp,
                    data=json.dumps(row, sort_keys=True).encode("utf-8"),
                )
            writer.finish()
        primary_path = mcap_path
        primary_format = "mcap"
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mcap_gap = f"mcap_unavailable:{type(exc).__name__}"
    index = {
        "schema_version": "policy_canary_telemetry_index.v1",
        "format": primary_format,
        "artifact": {
            "path": primary_path.name,
            "size_bytes": primary_path.stat().st_size,
            "sha256": _sha256(primary_path),
        },
        "schema": {
            "path": schema_path.name,
            "size_bytes": schema_path.stat().st_size,
            "sha256": _sha256(schema_path),
        },
        "channel_message_counts": channels,
        "message_count": len(rows),
        "attachments": [],
        "calibration_references": [
            (row.get("telemetry") or {}).get("camera_calibration") for row in rows
        ],
        "mcap_gap": mcap_gap,
        "evidence_gaps": sorted(
            {
                gap
                for row in rows
                for gap in (row.get("telemetry") or {}).get("evidence_gaps", [])
            }
        ),
        "index_digest": "",
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    index_path = output_root / "policy_canary_telemetry_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import mimetypes

    artifacts = []
    seen: set[str] = set()
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == PROVIDER_RESULT_FILENAME:
            continue
        relative = path.relative_to(output_root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        lowered = relative.lower()
        typed_evidence_role = next(
            (
                evidence_role
                for evidence_role in (
                    "reset_state",
                    "policy_query_receipt",
                    "action_sequence",
                    "action_delivery_readback",
                    "state_trace",
                    "contact_force_trace",
                    "task_object_trajectory",
                    "score_receipt",
                )
                if f".{evidence_role}.json" in lowered
            ),
            None,
        )
        role = (
            "indexed_episode_telemetry"
            if path in {primary_path, telemetry_path}
            else "telemetry_schema"
            if path == schema_path
            else "telemetry_index"
            if path == index_path
            else "review_video"
            if path.suffix.lower() in {".mp4", ".mov", ".webm"}
            else "lossless_frame_manifest"
            if "frame" in lowered and "manifest" in lowered
            else typed_evidence_role
            if typed_evidence_role is not None
            else "episode_evidence"
            if "episode" in lowered
            else "runtime_supporting_evidence"
        )
        artifacts.append(
            {
                "role": role,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return index, artifacts


def main() -> int:
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / PROVIDER_RESULT_FILENAME
    inputs = validate_runtime_input_manifest(
        _read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json")
    )
    authority = validate_session_authority(
        _read(runtime / "runtime_inputs" / "policy_canary_session_authority.json")
    )
    base_scene_plan = _read(
        runtime / "native_task_packet" / "native_task_arena_scene_plan.v1.json"
    )
    construction = _read(
        runtime
        / "runtime_inputs"
        / "native_task_arena_construction_result.v1.json"
    )
    if (
        construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("scene_plan_digest") != base_scene_plan.get("plan_digest")
        or construction.get("result_digest")
        != canonical_digest(construction, digest_field="result_digest")
    ):
        raise RuntimeError("policy_canary_construction_result_invalid")
    specs = {
        candidate: _read(
            runtime
            / "runtime_inputs"
            / f"policy_execution_spec.{candidate}.json"
        )
        for candidate in CANDIDATE_IDS
    }
    current_env: dict[str, Any] = {}

    def open_session(_inputs: Mapping[str, Any]) -> dict[str, Any]:
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        return {
            "simulation_app": simulation_app,
            "launch": launch,
            "provider_session_identity": _digest(launch),
        }

    def load_policy(_session: Mapping[str, Any], candidate: str) -> dict[str, Any]:
        from blueprint_pipeline.native_task_arena_policy_worker import (
            _policy_client,
            _runtime_groot_worker_identity,
        )

        spec = specs[candidate]
        groot_identity = None
        runtime_identity: Mapping[str, Any] = spec.get("runtime_identity") or {}
        if candidate == "groot_n17_droid":
            groot_identity, runtime_identity = _runtime_groot_worker_identity(
                output_root=output_root, spec=spec
            )
        client = _policy_client(
            spec, groot_worker_identity_receipt=groot_identity
        )
        return {
            "candidate_id": candidate,
            "client": client,
            "spec": spec,
            "checkpoint_digest": spec["checkpoint_digest"],
            "runtime_identity_digest": spec.get("runtime_identity_digest")
            or _digest(runtime_identity),
        }

    def run_episode(
        _session: Mapping[str, Any],
        policy: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        started_ns = time.time_ns()
        from blueprint_pipeline.adp009d_droid_action_execution import GripperConvention
        from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _gripper_convention_probe,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_policy_worker import (
            _PolicyQueryTracker,
            _to_tensor,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_episode_environment import (
            build_native_task_episode_environment,
        )
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
        )
        import torch

        scene_plan = _resolved_scene_plan(base_scene_plan, context)
        dependencies = preflight_native_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
        )
        if not dependencies["all_required_available"]:
            raise RuntimeError("policy_canary_dependency_preflight_failed")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not preconstruction["passed"]:
            raise RuntimeError("policy_canary_preconstruction_failed")
        built = build_native_task_arena_environment(
            scene_plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=runtime / "native_task_packet",
            preconstruction_receipt=preconstruction,
        )
        current_env["env"] = built.env
        device = read_native_task_arena_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not device["passed"]:
            raise RuntimeError("policy_canary_device_binding_failed")
        env = built.env
        seed = int(context["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        gripper = _gripper_convention_probe(
            env=env, robot=robot, seed=seed, torch=torch
        )
        if gripper["status"] != "measured":
            raise RuntimeError("policy_canary_gripper_unresolved")
        env.reset(seed=seed)
        servo = NativeFrankaDifferentialIkServo(
            env=env, robot=robot, gripper_convention=gripper
        )
        task_readback = (
            NativeArticulatedTaskArenaReadback(
                built,
                grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
            )
            if scene_plan["task_kind"] == "articulated_open_close"
            else None
        )
        episode_environment, environment_receipt = build_native_task_episode_environment(
            built=built,
            gripper_convention=gripper,
            servo=servo,
            task_readback=task_readback,
            to_tensor=_to_tensor,
        )
        tracker = _PolicyQueryTracker(policy["client"])
        spec = policy["spec"]
        episode_id = f"{authority['run_id']}--{context['cell_id']}--{context['candidate_id']}"
        try:
            episode = run_policy_episode(
                environment=episode_environment,
                policy=tracker,
                candidate_id=str(context["candidate_id"]),
                prompt=str(spec["prompt"]),
                task_spec=scene_plan["task_spec"],
                max_policy_queries=int(spec["max_policy_queries"]),
                settle_window_samples=int(
                    scene_plan["task_spec"]["settle_window_samples"]
                ),
                open_loop_horizon=int(spec["open_loop_horizon"]),
                gripper=GripperConvention(
                    closed_command=float(gripper["closed_command"]),
                    open_command=float(gripper["open_command"]),
                    measured_by_probe=True,
                ),
                media_output_dir=output_root / "episodes",
                episode_id=episode_id,
                scoring_authorized=True,
                require_complete_multicamera_media=True,
                require_prestart_readiness=True,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            current_env.clear()
        visual = episode.get("visual_evidence") or {}
        media = episode.get("media_artifacts") or {}
        motion = episode.get("motion_evidence") or {}
        telemetry = {
            "schema_version": "policy_canary_episode_telemetry.v1",
            "timebase": "unix_ns",
            "started_at_unix_ns": started_ns,
            "completed_at_unix_ns": time.time_ns(),
            "camera_calibration": episode.get("camera_calibration"),
            "policy_query_latency": episode.get("policy_query_latency"),
            "resource_telemetry": episode.get("resource_telemetry"),
            "channels": episode.get("telemetry_channels"),
            "wall_time_ns": time.time_ns() - started_ns,
            "evidence_gaps": [
                name
                for name, value in (
                    ("camera_calibration_unavailable", episode.get("camera_calibration")),
                    ("policy_query_latency_unavailable", episode.get("policy_query_latency")),
                    ("resource_telemetry_unavailable", episode.get("resource_telemetry")),
                    ("telemetry_channels_unavailable", episode.get("telemetry_channels")),
                )
                if value is None
            ],
            "mcap_gap": (
                None
                if episode.get("mcap_artifact")
                else "mcap_library_or_runtime_capture_unavailable"
            ),
        }
        media_artifacts = episode.get("media_artifacts") or []
        evidence_artifacts = {
            "frame_manifest": _bound_media_artifact(
                output_root,
                artifacts=media_artifacts,
                role="lossless_frame_manifest",
                role_match=lambda name: "frame_manifest" in name,
            ),
            "review_video": _bound_media_artifact(
                output_root,
                artifacts=media_artifacts,
                role="review_video",
                role_match=lambda name: "video" in name,
            ),
            "reset_state": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="reset_state",
                value=environment_receipt,
            ),
            "policy_query_receipt": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="policy_query_receipt",
                value={
                    "candidate_policy_queried": tracker.candidate_policy_queried,
                    "policy_queries": episode.get("policy_queries"),
                    "policy_query_latency": episode.get("policy_query_latency"),
                },
            ),
            "action_sequence": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="action_sequence",
                value=episode.get("commanded_actions"),
            ),
            "action_delivery_readback": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="action_delivery_readback",
                value=motion,
            ),
            "state_trace": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="state_trace",
                value=episode.get("state_trace"),
            ),
            "contact_force_trace": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="contact_force_trace",
                value=episode.get("contact_force_evidence"),
            ),
            "task_object_trajectory": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="task_object_trajectory",
                value=episode.get("task_object_trajectory"),
            ),
            "score_receipt": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="score_receipt",
                value=episode.get("score"),
            ),
        }
        return {
            "status": "completed",
            "candidate_policy_queried": tracker.candidate_policy_queried,
            "actions_reached_robot": bool(motion.get("actions_reached_robot")),
            "arm_moved": bool(motion.get("arm_moved")),
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "lossless_frame_manifest_digest": _digest(visual),
            "review_video_digest": _digest(media),
            "returned_action_sequence_digest": _digest(
                episode.get("commanded_actions")
            ),
            "action_delivery_readback_digest": _digest(motion),
            "state_trace_digest": _digest(episode.get("state_trace")),
            "contact_force_digest": _digest(episode.get("contact_force_evidence")),
            "task_object_trajectory_digest": _digest(
                episode.get("task_object_trajectory")
            ),
            "deterministic_score_digest": _digest(episode.get("score")),
            "scoring_authority": "deterministic_simulator_state",
            "episode": episode,
            "episode_environment": environment_receipt,
            "telemetry": telemetry,
            "telemetry_digest": _digest(telemetry),
            "code_identity_digest": authority.get("source_commit_digest")
            or authority["authority_digest"],
            "container_identity_digest": policy["runtime_identity_digest"],
            "scene_revision_digest": inputs.get("scene_revision_digest")
            or inputs["configuration_digest"],
            "scoring_version_digest": _digest("deterministic_simulator_state"),
            "evidence_artifacts": evidence_artifacts,
        }

    def close_policy(policy: Mapping[str, Any]) -> None:
        close = getattr(policy.get("client"), "close", None)
        if callable(close):
            close()

    def close_session(session: Mapping[str, Any]) -> dict[str, Any]:
        close = getattr(session.get("simulation_app"), "close", None)
        if callable(close):
            close()
        return {
            "status": "runtime_closed_pending_provider_teardown",
            "runtime_closed": True,
            "provider_closeout_pending": True,
        }

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=open_session,
        load_policy=load_policy,
        run_episode=run_episode,
        close_policy=close_policy,
        close_session=close_session,
        output_path=result_path,
        provider_closeout_pending=True,
    )
    telemetry_index, telemetry_artifacts = _write_indexed_telemetry(
        output_root, result["episodes"]
    )
    result["telemetry"] = telemetry_index
    result["artifact_inventory"] = telemetry_artifacts
    result["artifact_inventory_digest"] = _digest(telemetry_artifacts)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "runtime_completed_unqualified_pending_closeout" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
