"""Command adapter for OSCAR action-conditioned WAM rollout generation.

The adapter reads the Blueprint WAM rollout manifest path from
``BLUEPRINT_WAM_ROLLOUT_INPUT``, builds OSCAR's required first-frame plus
skeleton-conditioning inputs from MuJoCo review/trace artifacts, runs the
public OSCAR inference entrypoint, and writes Blueprint rollout JSON only when
OSCAR produces a generated MP4.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ADAPTER_ID = "blueprint_oscar_wam_command_adapter"
SCHEMA_VERSION = "oscar_wam_command_adapter.v1"
DEFAULT_NUM_FRAMES = 81
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640
DEFAULT_FPS = 15.0


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
            os.getenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_OSCAR_SOURCE_ROOT", ""),
        ]
    )


def _checkpoint_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_OSCAR_WAM_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", ""),
        ]
    )


def _selected_video_path(rollout_manifest: Mapping[str, Any]) -> Path:
    for row in rollout_manifest.get("selected_review_videos", []) or []:
        path = Path(_string(_mapping(row).get("path"))).expanduser()
        if path.is_file():
            return path.resolve()
    inputs = _mapping(rollout_manifest.get("inputs"))
    selection_manifest_path = Path(
        _string(inputs.get("review_video_selection_manifest"))
    ).expanduser()
    if selection_manifest_path.is_file():
        selection_manifest = _read_json(selection_manifest_path)
        for row in selection_manifest.get("selected_review_videos", []) or []:
            path = Path(_string(_mapping(row).get("path"))).expanduser()
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError("missing_selected_review_video")


def _task_prompt(rollout_manifest: Mapping[str, Any]) -> str:
    for row in rollout_manifest.get("task_prompts", []) or []:
        prompt = _string(_mapping(row).get("task_prompt"))
        if prompt:
            return prompt
    return "Predict the next robot-scene frames from Blueprint action conditioning."


def _trace_rows(rollout_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = _mapping(rollout_manifest.get("inputs"))
    trace_path = Path(_string(inputs.get("g1_mujoco_locomotion_trace_jsonl"))).expanduser()
    rows = _read_jsonl(trace_path)
    selected_episode = ""
    try:
        selected_video = _selected_video_path(rollout_manifest)
        selected_episode = selected_video.name.split("__", maxsplit=1)[0]
    except FileNotFoundError:
        selected_episode = ""
    if selected_episode:
        episode_rows = [row for row in rows if _string(row.get("episode_id")) == selected_episode]
        if episode_rows:
            return episode_rows
    return rows


def _sample_rows(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) == 1:
        return [dict(rows[0]) for _ in range(count)]
    sampled: list[dict[str, Any]] = []
    last = len(rows) - 1
    for index in range(count):
        source_index = round(index * last / max(count - 1, 1))
        sampled.append(dict(rows[source_index]))
    return sampled


def _point_from_root(row: Mapping[str, Any]) -> tuple[float, float, float]:
    position = row.get("root_position")
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)) and len(position) >= 3:
        return _number(position[0]), _number(position[1]), _number(position[2])
    return 0.0, 0.0, 0.8


def _screen_transform(rows: Sequence[Mapping[str, Any]], width: int, height: int) -> tuple[float, float, float, float]:
    points = [_point_from_root(row) for row in rows] or [(0.0, 0.0, 0.8)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 0.5)
    scale = min(width, height) * 0.46 / span
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    return center_x, center_y, scale, span


def _render_proxy_skeleton_video(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    import cv2
    import math
    import numpy as np

    sampled_rows = _sample_rows(trace_rows, num_frames)
    if not sampled_rows:
        raise ValueError("missing_locomotion_trace_for_oscar_skeleton_conditioning")
    center_x, center_y, scale, _span = _screen_transform(sampled_rows, width, height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2_video_writer_failed_for_oscar_skeleton_conditioning")

    action_counts: dict[str, int] = {}
    fall_count = 0
    for index, row in enumerate(sampled_rows):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        root_x, root_y, root_z = _point_from_root(row)
        sx = int(width * 0.5 + (root_y - center_y) * scale)
        sy = int(height * 0.72 - (root_x - center_x) * scale)
        sy -= int((root_z - 0.78) * scale * 0.2)
        yaw = _number(row.get("root_yaw_rad"))
        action = _mapping(row.get("active_action"))
        action_type = _string(action.get("action_type")) or "unknown"
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        if row.get("fall_detected") is True:
            fall_count += 1
        color = (240, 240, 240) if row.get("fall_detected") is not True else (60, 60, 255)
        accent = (90, 220, 255)
        torso = max(42, int(height * 0.105))
        head_radius = max(8, int(height * 0.021))
        shoulder_half = max(18, int(width * 0.033))
        hip_half = max(13, int(width * 0.024))
        leg = max(35, int(height * 0.086))
        arm = max(32, int(height * 0.078))
        swing = math.sin(index * 0.45)
        yaw_dx = int(math.sin(yaw) * 18)
        hip = (sx, sy)
        neck = (sx + yaw_dx, sy - torso)
        head = (neck[0], neck[1] - torso // 2)
        left_shoulder = (neck[0] - shoulder_half, neck[1] + 6)
        right_shoulder = (neck[0] + shoulder_half, neck[1] + 6)
        left_hip = (sx - hip_half, sy)
        right_hip = (sx + hip_half, sy)
        arm_reach = 1.0 if action_type == "manipulation_contact" else 0.55
        if action_type in {"waypoint", "base_velocity"}:
            arm_phase = swing
        elif action_type == "inspect_look":
            arm_phase = math.sin(index * 0.16)
        else:
            arm_phase = 0.0
        left_hand = (
            int(left_shoulder[0] - arm * 0.42),
            int(left_shoulder[1] + arm * (0.45 - 0.18 * arm_phase)),
        )
        right_hand = (
            int(right_shoulder[0] + arm * arm_reach),
            int(right_shoulder[1] + arm * (0.2 + 0.15 * arm_phase)),
        )
        left_foot = (int(left_hip[0] - leg * 0.22), int(left_hip[1] + leg * (1.0 + 0.08 * swing)))
        right_foot = (int(right_hip[0] + leg * 0.22), int(right_hip[1] + leg * (1.0 - 0.08 * swing)))
        for start, end in [
            (hip, neck),
            (left_shoulder, right_shoulder),
            (left_hip, right_hip),
            (left_shoulder, left_hand),
            (right_shoulder, right_hand),
            (left_hip, left_foot),
            (right_hip, right_foot),
        ]:
            cv2.line(frame, start, end, color, 4, cv2.LINE_AA)
        cv2.circle(frame, head, head_radius, color, -1, cv2.LINE_AA)
        if action_type in {"waypoint", "base_velocity", "manipulation_contact"}:
            vx = _number(action.get("vx_mps"))
            vy = _number(action.get("vy_mps"))
            arrow_end = (
                int(sx + vy * 100),
                int(sy - torso - vx * 100),
            )
            cv2.arrowedLine(frame, (sx, sy - torso), arrow_end, accent, 3, cv2.LINE_AA, tipLength=0.25)
        writer.write(frame)
    writer.release()
    return {
        "path": str(output_path),
        "frame_count": len(sampled_rows),
        "fps": fps,
        "width": width,
        "height": height,
        "conditioning_source": "blueprint_proxy_skeleton_from_mujoco_root_pose_and_endpoint_actions",
        "true_robot_proprioceptive_skeleton_available": False,
        "action_type_counts": [
            {"action_type": key, "count": action_counts[key]} for key in sorted(action_counts)
        ],
        "fall_frame_count": fall_count,
    }


def _extract_first_frame(
    *,
    review_video: Path,
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(review_video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("could_not_decode_selected_review_video_first_frame")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(output_path), resized)
    return {
        "path": str(output_path),
        "source_review_video_path": str(review_video),
        "width": width,
        "height": height,
    }


def _materialize_oscar_input_package(
    *,
    rollout_manifest: Mapping[str, Any],
    work_dir: Path,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
) -> dict[str, Any]:
    package_dir = work_dir / "oscar_input"
    review_video = _selected_video_path(rollout_manifest)
    trace_rows = _trace_rows(rollout_manifest)
    first_frame = _extract_first_frame(
        review_video=review_video,
        output_path=package_dir / "first_frame.png",
        width=width,
        height=height,
    )
    skeleton_video = _render_proxy_skeleton_video(
        trace_rows=trace_rows,
        output_path=package_dir / "blueprint_proxy_skeleton_conditioning.mp4",
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
    )
    manifest = {
        "schema_version": "blueprint_oscar_wam_input_package.v1",
        "status": "completed",
        "first_frame": first_frame,
        "skeleton_video": skeleton_video,
        "prompt": _task_prompt(rollout_manifest),
        "num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "source_review_video_path": str(review_video),
        "source_mujoco_endpoint_eval_job_dir": rollout_manifest.get(
            "source_mujoco_endpoint_eval_job_dir"
        ),
        "claim_boundary": {
            "skeleton_conditioning_is_proxy_from_mujoco_trace": True,
            "true_robot_proprioceptive_skeleton_available": False,
            "generated_input_is_not_model_output": True,
        },
    }
    _write_json(work_dir / "oscar_wam_input_package_manifest.json", manifest)
    return manifest


def _runtime_env(source_root: Path) -> dict[str, str]:
    pythonpath = os.pathsep.join(
        part
        for part in [
            str(_repo_src_root()),
            str(source_root),
            os.environ.get("PYTHONPATH", ""),
        ]
        if part
    )
    return {**os.environ, "PYTHONPATH": pythonpath}


def _run_import_probe(*, python: str, source_root: Path, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    if platform.system() == "Darwin" and not shutil.which("nvidia-smi"):
        return {
            "schema_version": "oscar_wam_runtime_import_probe.v1",
            "status": "blocked",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "module_available": {},
            "missing_modules": [],
            "torch_cuda_available": False,
            "platform_system": platform.system(),
            "blockers": [
                "blocked_oscar_linux_cuda_runtime_required",
                "blocked_oscar_requires_cuda_gpu_runtime",
            ],
            "stderr_size_bytes": 0,
            "stderr_omitted_to_avoid_secret_leakage": False,
        }
    probe = (
        "import importlib.util, json, platform\n"
        "mods=['torch','torchvision','cv2','decord','einops','diffusers','transformers','worldsim']\n"
        "available={m: bool(importlib.util.find_spec(m)) for m in mods}\n"
        "cuda=None\n"
        "try:\n"
        " import torch\n"
        " cuda=bool(torch.cuda.is_available())\n"
        "except Exception:\n"
        " cuda=False\n"
        "print(json.dumps({'module_available': available, 'torch_cuda_available': cuda, 'platform_system': platform.system()}))\n"
    )
    result = subprocess.run(
        [python, "-c", probe],
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    payload: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            value = json.loads(result.stdout)
            payload = dict(value) if isinstance(value, Mapping) else {}
        except json.JSONDecodeError:
            payload = {}
    module_available = _mapping(payload.get("module_available"))
    missing = [key for key, available in module_available.items() if available is False]
    blockers: list[str] = []
    if result.returncode != 0 or not module_available:
        blockers.append("blocked_oscar_runtime_import_probe_failed")
    if missing:
        blockers.append("blocked_missing_oscar_runtime_imports")
    if payload.get("torch_cuda_available") is not True:
        blockers.append("blocked_oscar_requires_cuda_gpu_runtime")
    return {
        "schema_version": "oscar_wam_runtime_import_probe.v1",
        "status": "completed" if not blockers else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "module_available": module_available,
        "missing_modules": missing,
        "torch_cuda_available": payload.get("torch_cuda_available"),
        "platform_system": payload.get("platform_system"),
        "blockers": blockers,
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
    }


def _redacted_argv(argv: Sequence[str], checkpoint: Path) -> list[str]:
    checkpoint_text = str(checkpoint)
    return ["<checkpoint_path_configured>" if item == checkpoint_text else item for item in argv]


def _run_oscar(
    *,
    python: str,
    source_root: Path,
    checkpoint: Path,
    package_manifest: Mapping[str, Any],
    output_video: Path,
    timeout_seconds: float,
    num_steps: int,
    guidance: float,
    seed: int,
) -> dict[str, Any]:
    started = time.monotonic()
    argv = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "inference/inference_oscar.py",
        "--checkpoint",
        str(checkpoint),
        "--first-frame",
        _string(_mapping(package_manifest.get("first_frame")).get("path")),
        "--skeleton-video",
        _string(_mapping(package_manifest.get("skeleton_video")).get("path")),
        "--start-frame",
        "0",
        "--prompt",
        _string(package_manifest.get("prompt")),
        "--num-steps",
        str(num_steps),
        "--guidance",
        str(guidance),
        "--seed",
        str(seed),
        "--num-frames",
        str(int(package_manifest.get("num_frames") or DEFAULT_NUM_FRAMES)),
        "--height",
        str(int(package_manifest.get("height") or DEFAULT_HEIGHT)),
        "--width",
        str(int(package_manifest.get("width") or DEFAULT_WIDTH)),
        "--fps",
        str(float(package_manifest.get("fps") or DEFAULT_FPS)),
        "--output",
        str(output_video),
    ]
    result = subprocess.run(
        argv,
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    return {
        "schema_version": "oscar_wam_subprocess_result.v1",
        "status": "completed" if result.returncode == 0 else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "argv_redacted": _redacted_argv(argv, checkpoint),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": [] if result.returncode == 0 else ["oscar_inference_command_nonzero"],
    }


def _rollout_payload(
    *,
    package_manifest: Mapping[str, Any],
    checkpoint: Path,
    source_root: Path,
    subprocess_detail: Mapping[str, Any],
    output_video: Path,
) -> dict[str, Any]:
    video_exists = output_video.is_file()
    rollouts = [
        {
            "rollout_id": "oscar_wam_rollout_0001",
            "policy_id": ADAPTER_ID,
            "model_candidate": "oscar_wam",
            "generated_video_path": str(output_video),
            "source_review_video_path": package_manifest.get("source_review_video_path"),
            "model_rollout_confidence": None,
            "generated_rollout_termination_reason": "oscar_inference_command_completed",
            "success_label_source": "generated_video_requires_review",
        }
    ] if video_exists else []
    blockers = [] if rollouts else ["blocked_no_generated_oscar_mp4"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if rollouts else "blocked",
        "adapter_id": ADAPTER_ID,
        "rollouts": rollouts,
        "generated_video_count": len(rollouts),
        "model_provenance": {
            "candidate": "oscar_wam",
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "oscar_public_inference_entrypoint": str(source_root / "inference" / "inference_oscar.py"),
        },
        "input_package": dict(package_manifest),
        "oscar_subprocess": dict(subprocess_detail),
        "blockers": blockers,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", default=os.getenv("BLUEPRINT_OSCAR_WAM_PYTHON") or sys.executable)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--num-frames", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", str(DEFAULT_NUM_FRAMES))))
    parser.add_argument("--height", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_HEIGHT", str(DEFAULT_HEIGHT))))
    parser.add_argument("--width", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_WIDTH", str(DEFAULT_WIDTH))))
    parser.add_argument("--fps", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_FPS", str(DEFAULT_FPS))))
    parser.add_argument("--num-steps", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "35")))
    parser.add_argument("--guidance", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_GUIDANCE", "6.0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("BLUEPRINT_OSCAR_WAM_SEED", "42")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("BLUEPRINT_OSCAR_WAM_TIMEOUT_SECONDS", "3600")))
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
    output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_path.parent / "oscar_wam_command_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    if source_root is None:
        blockers.append("blocked_missing_oscar_source_root")
    elif not (source_root / "inference" / "inference_oscar.py").is_file():
        blockers.append("blocked_missing_oscar_inference_entrypoint")
    if checkpoint is None:
        blockers.append("blocked_missing_oscar_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_configured_oscar_checkpoint_path_missing")
    if not shutil.which(args.python) and not Path(args.python).expanduser().is_file():
        blockers.append("blocked_configured_python_missing")

    package_manifest: dict[str, Any] = {}
    if not blockers:
        try:
            rollout_input = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).expanduser().resolve()
            rollout_manifest = _read_json(rollout_input)
            package_manifest = _materialize_oscar_input_package(
                rollout_manifest=rollout_manifest,
                work_dir=work_dir,
                width=args.width,
                height=args.height,
                fps=args.fps,
                num_frames=args.num_frames,
            )
        except Exception as exc:
            blockers.append(f"blocked_oscar_input_package_materialization_failed:{type(exc).__name__}")

    if blockers:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": blockers,
            "source_root": str(source_root) if source_root else None,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "input_package": package_manifest or None,
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
    _write_json(work_dir / "oscar_wam_import_probe.json", probe)
    if args.probe_only or probe["status"] != "completed":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": probe["status"],
            "adapter_id": ADAPTER_ID,
            "probe_only": bool(args.probe_only),
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "input_package": package_manifest,
            "import_probe": probe,
            "blockers": probe.get("blockers", []),
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
        return payload

    output_video = work_dir / "oscar_generated_rollout.mp4"
    subprocess_detail = _run_oscar(
        python=args.python,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_video=output_video,
        timeout_seconds=args.timeout_seconds,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
    )
    payload = _rollout_payload(
        package_manifest=package_manifest,
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail=subprocess_detail,
        output_video=output_video,
    )
    if subprocess_detail["status"] != "completed" and not payload["rollouts"]:
        payload["status"] = "blocked"
        payload["blockers"] = list(subprocess_detail.get("blockers") or payload["blockers"])
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")).resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "adapter_id": ADAPTER_ID,
            "blockers": [f"oscar_wam_adapter_exception:{type(exc).__name__}"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
        _write_json(output_path, payload)
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
