"""Command adapter for OSCAR/Cosmos action-conditioned WAM rollout generation.

The adapter is the process boundary used by ``oscar_cosmos_wam_evaluator``.
It reads the Blueprint WAM rollout manifest path from environment variables,
materializes a Cosmos Predict2.5 action-conditioned input package, runs the
official Cosmos action-conditioned entrypoint, and writes Blueprint rollout
JSON only when a generated MP4 exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_space_registry import SC3_7D_DELTA_EE, get_action_space
from .wam_generated_video_review import validate_generated_mp4_for_review


ADAPTER_ID = "blueprint_oscar_cosmos_action_conditioned_wam_adapter"
SCHEMA_VERSION = "oscar_cosmos_wam_command_adapter.v1"
DEFAULT_EXPERIMENT = "ac_reason_embeddings_rectified_flow_2b_256_320"
DEFAULT_MODEL = "2B/robot/action-cond"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _first_existing_path(paths: Sequence[str]) -> Path | None:
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _source_root_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_OSCAR_COSMOS_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_COSMOS_WAM_SOURCE_ROOT", ""),
        ]
    )


def _checkpoint_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_COSMOS_WAM_CHECKPOINT", ""),
        ]
    )


def _preference_list(env_name: str, default: Sequence[str]) -> list[str]:
    configured = os.getenv(env_name, "")
    values = configured.split(",") if configured else list(default)
    return [_string(value) for value in values if _string(value)]


def _video_candidates(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rollout_manifest.get("selected_review_videos", []) or []:
        if isinstance(row, Mapping):
            rows.append(dict(row))
    inputs = _mapping(rollout_manifest.get("inputs"))
    selection_manifest_path = Path(
        _string(inputs.get("review_video_selection_manifest"))
    ).expanduser()
    if selection_manifest_path.is_file():
        selection_manifest = _read_json(selection_manifest_path)
        for row in selection_manifest.get("selected_review_videos", []) or []:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _task_prompt_by_run_id(rollout_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in rollout_manifest.get("task_prompts", []) or []:
        if isinstance(row, Mapping) and _string(row.get("scenario_eval_run_id")):
            rows[_string(row.get("scenario_eval_run_id"))] = dict(row)
    return rows


def _selected_video_row(rollout_manifest: Mapping[str, Any]) -> dict[str, Any]:
    camera_preferences = _preference_list(
        "BLUEPRINT_WAM_PREFERRED_CAMERA",
        ("robot_pov", "torso_pov", "robot_follow", "third_person", "overhead"),
    )
    task_preferences = _preference_list("BLUEPRINT_WAM_PREFERRED_TASK_ID", ())
    prompt_rows = _task_prompt_by_run_id(rollout_manifest)
    ranked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for index, candidate in enumerate(_video_candidates(rollout_manifest)):
        path = Path(_string(candidate.get("path"))).expanduser()
        if not path.is_file():
            continue
        row = dict(candidate)
        run_id = _string(row.get("scenario_eval_run_id"))
        if run_id and run_id in prompt_rows:
            row = {**prompt_rows[run_id], **row}
        text = " ".join(
            _string(row.get(key))
            for key in ("task_id", "episode_id", "scenario_eval_run_id", "path")
        )
        camera = _string(row.get("camera"))
        camera_rank = camera_preferences.index(camera) if camera in camera_preferences else len(camera_preferences)
        task_rank = len(task_preferences)
        for pref_index, task_id in enumerate(task_preferences):
            if task_id and task_id in text:
                task_rank = pref_index
                break
        ranked.append(((task_rank, camera_rank, index), {**row, "path": str(path.resolve())}))
    if ranked:
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]
    raise FileNotFoundError("missing_selected_review_video")


def _selected_video_path(rollout_manifest: Mapping[str, Any]) -> Path:
    return Path(_selected_video_row(rollout_manifest)["path"]).resolve()


def _task_prompt(rollout_manifest: Mapping[str, Any]) -> str:
    try:
        selected = _selected_video_row(rollout_manifest)
        prompt = _string(selected.get("task_prompt"))
        if prompt:
            return prompt
    except FileNotFoundError:
        pass
    for row in rollout_manifest.get("task_prompts", []) or []:
        prompt = _string(_mapping(row).get("task_prompt"))
        if prompt:
            return prompt
    return "Predict the next robot-scene frames from Blueprint action conditioning."


def _action_trace_rows(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = _mapping(rollout_manifest.get("inputs"))
    trace_path = Path(_string(inputs.get("normalized_policy_action_trace_jsonl"))).expanduser()
    return _read_jsonl(trace_path)


def _action_vector(row: Mapping[str, Any]) -> list[float]:
    normalized = _mapping(row.get("normalized_action"))
    candidates = (
        row.get("delta_end_effector_pose_7d"),
        row.get("sc3_7d_delta_end_effector_pose"),
        row.get("action_vector_7d"),
        normalized.get("delta_end_effector_pose_7d"),
        normalized.get("sc3_7d_delta_end_effector_pose"),
        normalized.get("action_vector_7d"),
    )
    vector: Sequence[Any] | None = None
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            vector = candidate
            break
    if vector is None:
        raise RuntimeError("cosmos_action_trace_missing_explicit_sc3_7d_action")
    # The Cosmos WAM lane is bound to one registered action space rather than a
    # bare literal, so widening it to another embodiment is a registry change
    # with its own contract rather than an edit to a magic number here.
    space = get_action_space(SC3_7D_DELTA_EE)
    if len(vector) != space.dim:
        raise RuntimeError(f"cosmos_action_trace_{space.dim_blocker}")
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool):
            raise RuntimeError("cosmos_action_trace_action_non_numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cosmos_action_trace_action_non_numeric") from exc
        if not math.isfinite(number):
            raise RuntimeError("cosmos_action_trace_action_non_finite")
        values.append(number)
    return values


def _action_sequence(rows: Sequence[Mapping[str, Any]], *, chunk_size: int) -> list[list[float]]:
    required = max(chunk_size, 1)
    if len(rows) < required:
        raise RuntimeError("cosmos_action_trace_has_fewer_actions_than_requested_chunk")
    return [_action_vector(row) for row in rows[:required]]


def _materialize_cosmos_input_package(
    *,
    rollout_manifest: Mapping[str, Any],
    work_dir: Path,
    chunk_size: int,
    resolution: str,
    guidance: int,
    num_steps: int,
) -> dict[str, Any]:
    input_root = work_dir / "cosmos_input" / "bridge"
    annotation_dir = input_root / "annotation" / "test"
    video_dir = input_root / "videos" / "test" / "blueprint_0"
    save_root = work_dir / "cosmos_generated"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)

    selected_video = _selected_video_row(rollout_manifest)
    review_video = Path(selected_video["path"]).resolve()
    local_video = video_dir / "rgb.mp4"
    if review_video.resolve() != local_video.resolve():
        shutil.copyfile(review_video, local_video)

    source_action_rows = _action_trace_rows(rollout_manifest)
    actions = _action_sequence(source_action_rows, chunk_size=chunk_size)
    action_records = [
        {
            "action_id": f"action-{index:02d}",
            "action_sha256": hashlib.sha256(
                json.dumps(action, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "action_vector_7d": action,
        }
        for index, action in enumerate(actions)
    ]
    annotation = {
        "schema_version": "blueprint_cosmos_action_conditioning_annotation.v1",
        "task": "blueprint_mujoco_unitree_g1_action_conditioned_rollout",
        "texts": [_task_prompt(rollout_manifest)],
        "videos": [{"video_path": "videos/test/blueprint_0/rgb.mp4"}],
        "action": actions,
        "blueprint_source": {
            "source_mujoco_endpoint_eval_job_dir": rollout_manifest.get(
                "source_mujoco_endpoint_eval_job_dir"
            ),
            "selected_review_video_path": str(review_video),
            "selected_review_video": selected_video,
            "selected_camera": selected_video.get("camera"),
            "scenario_eval_run_id": selected_video.get("scenario_eval_run_id"),
            "task_id": selected_video.get("task_id"),
            "spawn_id": selected_video.get("spawn_id"),
            "action_trace_jsonl": _mapping(rollout_manifest.get("inputs")).get(
                "normalized_policy_action_trace_jsonl"
            ),
        },
    }
    annotation_path = annotation_dir / "0.json"
    _write_json(annotation_path, annotation)

    inference_params = {
        "name": "blueprint_mujoco_unitree_g1_action_conditioned_rollout",
        "input_root": str(input_root),
        "input_json_sub_folder": "annotation/test",
        "save_root": str(save_root),
        "guidance": guidance,
        "resolution": resolution,
        "camera_id": 0,
        "start": 0,
        "end": 1,
        "fps_downsample_ratio": 1,
        "gripper_scale": 1.0,
        "gripper_key": "continuous_gripper_state",
        "state_key": "state",
        "chunk_size": chunk_size,
        "reverse": False,
        "single_chunk": True,
        "start_frame_idx": 0,
        "save_fps": 20,
        "num_latent_conditional_frames": 1,
        "num_steps": num_steps,
        "action_scaler": 1.0,
        "use_quat": False,
        "action_load_fn": (
            "blueprint_pipeline.oscar_cosmos_wam_command_adapter."
            "load_blueprint_action_fn"
        ),
        "negative_prompt": (
            "low quality, static scene, no robot motion, incoherent physics, "
            "visual artifacts, broken contact geometry"
        ),
        "seed": 0,
        "prompt": _task_prompt(rollout_manifest),
    }
    inference_path = work_dir / "cosmos_inference_params.json"
    _write_json(inference_path, inference_params)
    manifest = {
        "schema_version": "blueprint_cosmos_rollout_input_package.v1",
        "status": "completed",
        "input_root": str(input_root),
        "annotation_path": str(annotation_path),
        "inference_params_path": str(inference_path),
        "save_root": str(save_root),
        "source_review_video_path": str(review_video),
        "source_review_video": selected_video,
        "source_camera": selected_video.get("camera"),
        "scenario_eval_run_id": selected_video.get("scenario_eval_run_id"),
        "task_id": selected_video.get("task_id"),
        "spawn_id": selected_video.get("spawn_id"),
        "action_count": len(actions),
        "action_records": action_records,
        "control_rate_hz": 20.0,
        "chunk_start_timestamp_sec": 0.0,
        "action_load_fn": inference_params["action_load_fn"],
    }
    _write_json(work_dir / "cosmos_rollout_input_package_manifest.json", manifest)
    return manifest


def _redacted_argv(argv: Sequence[str], checkpoint: Path) -> list[str]:
    redacted: list[str] = []
    checkpoint_value = str(checkpoint)
    for item in argv:
        redacted.append("<checkpoint_path_configured>" if item == checkpoint_value else item)
    return redacted


def _runtime_env(source_root: Path) -> dict[str, str]:
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(_repo_src_root()), str(source_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_parts)}


def _run_import_probe(*, python: str, source_root: Path, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "import json, importlib.util; "
                "mods=['cosmos_oss','cosmos_predict2','tyro','torch']; "
                "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))"
            ),
        ],
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    available: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            available = json.loads(result.stdout)
        except json.JSONDecodeError:
            available = {}
    missing = [name for name, present in available.items() if not present]
    return {
        "schema_version": "oscar_cosmos_runtime_import_probe.v1",
        "status": "completed" if result.returncode == 0 and not missing else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "module_available": available,
        "blockers": []
        if result.returncode == 0 and not missing
        else ["blocked_missing_cosmos_runtime_import"],
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
    }


def _failure_signals(*values: str) -> list[str]:
    text = "\n".join(value for value in values if value).lower()
    signals: list[str] = []
    if "cuda extra not installed" in text or "uv sync --extra=" in text:
        signals.append("blocked_cosmos_cuda_extra_not_installed")
    if "cuda" in text and ("not available" in text or "no cuda" in text):
        signals.append("blocked_cuda_not_available")
    if "mps" in text and ("not available" in text or "not supported" in text):
        signals.append("blocked_mps_not_available")
    if "out of memory" in text or "memoryerror" in text:
        signals.append("blocked_model_runtime_out_of_memory")
    if "no module named" in text or "modulenotfounderror" in text:
        signals.append("blocked_missing_python_module")
    if "checkpoint" in text and ("not found" in text or "missing" in text):
        signals.append("blocked_checkpoint_load_failed")
    return signals


def _run_cosmos(
    *,
    python: str,
    source_root: Path,
    checkpoint: Path,
    package_manifest: Mapping[str, Any],
    output_dir: Path,
    model: str,
    experiment: str,
    context_parallel_size: int,
    timeout_seconds: float,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    entrypoint = source_root / "examples" / "action_conditioned.py"
    inference_params = Path(_string(package_manifest.get("inference_params_path")))
    argv = [
        python,
        str(entrypoint),
        "-i",
        str(inference_params),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--experiment",
        experiment,
        "--model",
        model,
        "--context-parallel-size",
        str(context_parallel_size),
    ]
    argv.extend(extra_args)
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=str(source_root),
            env=_runtime_env(source_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": "oscar_cosmos_subprocess_result.v1",
            "status": "blocked",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "argv_redacted": _redacted_argv(argv, checkpoint),
            "stdout_size_bytes": len(exc.stdout or ""),
            "stderr_size_bytes": len(exc.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(exc.stderr),
            "blockers": ["cosmos_action_conditioned_command_timeout"],
        }
    failure_signals = _failure_signals(result.stdout or "", result.stderr or "")
    blockers = [] if result.returncode == 0 else ["cosmos_action_conditioned_command_nonzero"]
    blockers.extend(signal for signal in failure_signals if signal not in blockers)
    return {
        "schema_version": "oscar_cosmos_subprocess_result.v1",
        "status": "completed" if result.returncode == 0 else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "argv_redacted": _redacted_argv(argv, checkpoint),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": blockers,
    }


def _rollout_payload(
    *,
    package_manifest: Mapping[str, Any],
    checkpoint: Path,
    source_root: Path,
    subprocess_detail: Mapping[str, Any],
    model: str,
    experiment: str,
) -> dict[str, Any]:
    save_root = Path(_string(package_manifest.get("save_root")))
    generated_videos = sorted(path.resolve() for path in save_root.rglob("*.mp4"))
    video_validations = [
        validate_generated_mp4_for_review(path) for path in generated_videos
    ]
    rollouts = []
    for index, (path, validation) in enumerate(zip(generated_videos, video_validations), start=1):
        if validation.get("status") != "completed":
            continue
        rollouts.append(
            {
                "rollout_id": f"oscar_cosmos_rollout_{index:04d}",
                "policy_id": ADAPTER_ID,
                "model_candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE") or "oscar_wam",
                "model": model,
                "experiment": experiment,
                "generated_video_path": str(path),
                "source_review_video_path": package_manifest.get("source_review_video_path"),
                "source_camera": package_manifest.get("source_camera"),
                "scenario_eval_run_id": package_manifest.get("scenario_eval_run_id"),
                "task_id": package_manifest.get("task_id"),
                "spawn_id": package_manifest.get("spawn_id"),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "cosmos_command_completed",
                "success_label_source": "generated_video_requires_review",
                "generated_video_review_validation": validation,
            }
        )
    status = "completed" if rollouts else "blocked"
    validation_blockers = sorted(
        {
            str(blocker)
            for validation in video_validations
            for blocker in validation.get("blockers", [])
            if str(blocker)
        }
    )
    blockers = [] if rollouts else [
        "blocked_generated_cosmos_mp4_not_reviewable"
        if generated_videos
        else "blocked_no_generated_cosmos_mp4",
        *validation_blockers,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "adapter_id": ADAPTER_ID,
        "rollouts": rollouts,
        "generated_video_count": len(generated_videos),
        "generated_video_review_validations": video_validations,
        "model_provenance": {
            "candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE") or "oscar_wam",
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "model": model,
            "experiment": experiment,
        },
        "input_package": dict(package_manifest),
        "cosmos_subprocess": dict(subprocess_detail),
        "blockers": blockers,
        "fresh_model_command_executed_this_invocation": bool(
            rollouts and subprocess_detail.get("status") == "completed"
        ),
        "fresh_model_run_claimed": bool(
            rollouts and subprocess_detail.get("status") == "completed"
        ),
        "learned_wam_model_ran": bool(
            rollouts and subprocess_detail.get("status") == "completed"
        ),
        "truth_boundary": {
            "generated_video_is_model_output": bool(
                rollouts and subprocess_detail.get("status") == "completed"
            ),
            "generated_rollout_not_physical_robot_proof": True,
            "generated_success_label_requires_external_vlm_or_human_judge": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def load_blueprint_action_fn():
    """Cosmos action loader that consumes Blueprint's generated annotation."""

    def load_fn(json_data: dict, video_path: str, args: Any) -> dict[str, Any]:
        import mediapy
        import numpy as np

        actions = np.asarray(json_data.get("action") or [], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError("Blueprint Cosmos action annotations must be shaped [N, 7]")
        video_array = mediapy.read_video(video_path)
        start_frame = int(getattr(args, "start_frame_idx", 0) or 0)
        start_frame = max(0, min(start_frame, len(video_array) - 1))
        img_array = video_array[start_frame]
        resolution = getattr(args, "resolution", "none")
        if resolution and resolution != "none":
            height, width = [int(part) for part in str(resolution).split(",", maxsplit=1)]
            img_array = mediapy.resize_image(img_array, (height, width))
        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
        }

    return load_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--python",
        default=os.getenv("BLUEPRINT_OSCAR_COSMOS_PYTHON") or sys.executable,
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--model",
        default=os.getenv("BLUEPRINT_OSCAR_COSMOS_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("BLUEPRINT_OSCAR_COSMOS_EXPERIMENT") or DEFAULT_EXPERIMENT,
    )
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("BLUEPRINT_OSCAR_COSMOS_CHUNK_SIZE", "12")),
    )
    parser.add_argument(
        "--resolution",
        default=os.getenv("BLUEPRINT_OSCAR_COSMOS_RESOLUTION") or "256,320",
    )
    parser.add_argument(
        "--guidance",
        type=int,
        default=int(os.getenv("BLUEPRINT_OSCAR_COSMOS_GUIDANCE", "0")),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.getenv("BLUEPRINT_OSCAR_COSMOS_NUM_STEPS", "35")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("BLUEPRINT_OSCAR_COSMOS_TIMEOUT_SECONDS", "3600")),
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--probe-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    source_root = (
        args.source_root.expanduser().resolve() if args.source_root else _source_root_from_env()
    )
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else _checkpoint_from_env()
    )
    output_path = Path(
        os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
    ).resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_path.parent / "oscar_cosmos_command_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    if source_root is None:
        blockers.append("blocked_missing_oscar_cosmos_source_root")
    elif not (source_root / "examples" / "action_conditioned.py").is_file():
        blockers.append("blocked_missing_cosmos_action_conditioned_entrypoint")
    if checkpoint is None:
        blockers.append("blocked_missing_oscar_cosmos_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_configured_oscar_cosmos_checkpoint_path_missing")
    if not shutil.which(args.python) and not Path(args.python).expanduser().is_file():
        blockers.append("blocked_configured_python_missing")

    if blockers:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": blockers,
            "source_root": str(source_root) if source_root else None,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload

    assert source_root is not None
    assert checkpoint is not None
    probe = _run_import_probe(
        python=args.python,
        source_root=source_root,
        timeout_seconds=min(args.timeout_seconds, 120.0),
    )
    _write_json(work_dir / "oscar_cosmos_import_probe.json", probe)
    if args.probe_only:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": probe["status"],
            "adapter_id": ADAPTER_ID,
            "probe_only": True,
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "import_probe": probe,
            "blockers": probe.get("blockers", []),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload
    if probe["status"] != "completed":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "import_probe": probe,
            "blockers": probe.get("blockers", []),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload

    rollout_input = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).expanduser().resolve()
    rollout_manifest = _read_json(rollout_input)
    package_manifest = _materialize_cosmos_input_package(
        rollout_manifest=rollout_manifest,
        work_dir=work_dir,
        chunk_size=args.chunk_size,
        resolution=args.resolution,
        guidance=args.guidance,
        num_steps=args.num_steps,
    )
    cosmos_output_dir = work_dir / "cosmos_output"
    subprocess_detail = _run_cosmos(
        python=args.python,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=cosmos_output_dir,
        model=args.model,
        experiment=args.experiment,
        context_parallel_size=args.context_parallel_size,
        timeout_seconds=args.timeout_seconds,
        extra_args=[item for value in args.extra_arg for item in shlex.split(value)],
    )
    payload = _rollout_payload(
        package_manifest=package_manifest,
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail=subprocess_detail,
        model=args.model,
        experiment=args.experiment,
    )
    if subprocess_detail["status"] != "completed" and not payload["rollouts"]:
        payload["status"] = "blocked"
        payload["blockers"] = list(subprocess_detail.get("blockers") or [])
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(
            os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
        ).resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": [f"oscar_cosmos_adapter_exception:{type(exc).__name__}"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
