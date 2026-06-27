"""Per-step closed-loop eval: policy <-> OSCAR-2B WAM <-> SAM3/DA3 perception harness.

This is the true MuJoCo-parity loop (mujoco_g1_wam_vla_policy_endpoint_eval lines 4513-4559)
applied to the Isaac G1 lane, instead of the one-shot batch OSCAR clip the provider lane
generates. Each step:

  1. the policy emits an action (the deterministic G1 walk-to-target controller),
  2. the WAM generates the NEXT observation conditioned on that action (pluggable backend:
     a local stub for hermetic tests, real OSCAR-2B per-step for the GPU run),
  3. the SAM3/DA3 perception harness analyses that generated frame right away
     (run_wam_derived_observation_harness_step, real backend on GPU or fixture locally),
  4. the derived observations feed forward into the next step.

The WAM generation and the harness backend are both injected, so the loop structure can be
validated end-to-end with zero GPU spend and the same code path then runs the real backends.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .isaac_g1_policy import (
    DeterministicWalkToTargetPolicy,
    StepContext,
    action_record,
    interpolate_route,
)
from .wam_derived_observation_harness import run_wam_derived_observation_harness_step

LOOP_SCHEMA_VERSION = "oscar_isaac_closed_loop_eval.v1"

# A WAM generation backend: given the current observation frame, the policy action, the step
# index, and the action history, produce the next-observation frame path (and optional video).
# Returns a mapping with at least {"generated_frame_path": <path>}.
WamGenerateNext = Callable[
    [str, Mapping[str, Any], int, Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _policy_observation(frame_path: str, target: Sequence[float], step_index: int) -> dict[str, Any]:
    """The policy observation handed to the harness for this step (evaluator-controlled state
    kept separate from the WAM's pixel-inferred fields)."""
    return {
        "schema_version": "oscar_isaac_closed_loop_observation.v1",
        "step_index": step_index,
        "visual_observation": {"generated_frame_path": _string(frame_path)},
        "task_target_position_xyz": [round(float(c), 6) for c in target],
    }


def build_oscar_per_step_request(
    *,
    current_frame_path: str,
    action: Mapping[str, Any],
    step_index: int,
    task_prompt: str,
    num_frames: int,
    output_dir: str | Path,
    skeleton_landmarks: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Shape one per-step OSCAR-2B next-observation generation request.

    OSCAR generates a short clip forward from the current observation (``current_frame_path``),
    conditioned on the task prompt and the projected G1 skeleton for this step's action. The
    NEXT observation is the last frame of that clip. This is pure request shaping with no GPU or
    OSCAR import, so it is fully unit-testable; the actual inference is the injected callable in
    :func:`make_oscar_per_step_wam_backend`.
    """
    return {
        "schema_version": "oscar_per_step_generation_request.v1",
        "step_index": int(step_index),
        "reference_frame_path": _string(current_frame_path),
        "task_prompt": _string(task_prompt),
        "num_frames": max(1, int(num_frames)),
        "seed": int(seed) + int(step_index),
        "output_dir": str(Path(output_dir).expanduser() / f"oscar_step_{step_index:04d}"),
        "policy_action": dict(action),
        "root_position": list(action.get("root_position") or []),
        "root_yaw_radians": action.get("root_yaw_radians"),
        "projected_landmark_count": len(skeleton_landmarks or []),
        "skeleton_landmarks": [dict(landmark) for landmark in (skeleton_landmarks or [])],
    }


def make_oscar_per_step_wam_backend(
    *,
    oscar_generate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    work_dir: str | Path,
    task_prompt: str,
    num_frames: int = 8,
    skeleton_for_action: Callable[[Mapping[str, Any], int], Sequence[Mapping[str, Any]]] | None = None,
    seed: int = 42,
) -> WamGenerateNext:
    """A ``wam_generate_next`` backend that drives real per-step OSCAR-2B generation.

    ``oscar_generate`` is the injected inference call — it receives a per-step request (see
    :func:`build_oscar_per_step_request`) and must return a mapping with a ``generated_frame_path``
    (the next observation) and optionally ``generated_video_path``. On GPU this is a thin call into
    a persistent OSCAR-2B pod; in tests it is mocked. ``skeleton_for_action`` projects the G1
    skeleton landmarks for an action (the Isaac projector at run time; ``None`` omits conditioning).
    """
    resolved_work = Path(work_dir).expanduser().resolve()

    def _generate_next(
        current_frame: str,
        action: Mapping[str, Any],
        step_index: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        skeleton = list(skeleton_for_action(action, step_index)) if skeleton_for_action else []
        request = build_oscar_per_step_request(
            current_frame_path=current_frame,
            action=action,
            step_index=step_index,
            task_prompt=task_prompt,
            num_frames=num_frames,
            output_dir=resolved_work,
            skeleton_landmarks=skeleton,
            seed=seed,
        )
        result = dict(oscar_generate(request) or {})
        generated_frame = _string(result.get("generated_frame_path"))
        return {
            "generated_frame_path": generated_frame,
            "generated_video_path": result.get("generated_video_path"),
            "skeleton_conditioning": {"landmarks": skeleton} if skeleton else None,
            "wam_backend": "oscar_2b_per_step",
            "wam_generation_status": result.get("status"),
            "wam_generation_blockers": list(result.get("blockers") or []),
        }

    return _generate_next


def extract_last_frame_via_opencv(video_path: str | Path, out_dir: str | Path) -> Path | None:
    """Default ``extract_next_frame``: the last frame of an OSCAR clip is the next observation.

    Uses OpenCV so it works on the pod next to OSCAR. Returns the saved PNG path or None.
    """
    import cv2  # local import: only needed where a real clip is produced (the GPU pod)

    resolved_out = Path(out_dir).expanduser()
    resolved_out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
    finally:
        capture.release()
    if last_frame is None:
        return None
    frame_path = resolved_out / "next_observation.png"
    if not cv2.imwrite(str(frame_path), last_frame):
        return None
    return frame_path


def build_oscar_inference_argv(
    *,
    python: str,
    oscar_repo: str | Path,
    checkpoint: str | Path,
    first_frame_path: str,
    prompt: str,
    num_frames: int,
    num_steps: int,
    guidance: float,
    seed: int,
    height: int,
    width: int,
    fps: float,
    output_video: str | Path,
    skeleton_video: str | Path | None = None,
) -> list[str]:
    """The OSCAR inference argv for one per-step next-observation generation.

    Mirrors oscar_wam_command_adapter's invocation: torch.distributed.run inference_oscar.py with
    the current observation as --first-frame and (optionally) the action's projected skeleton as
    --skeleton-video. Pure argv construction so the real backend below stays unit-testable.
    """
    repo = Path(oscar_repo).expanduser()
    argv = [
        python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        str(repo / "inference" / "inference_oscar.py"),
        "--checkpoint",
        str(checkpoint),
        "--first-frame",
        _string(first_frame_path),
        "--start-frame",
        "0",
        "--prompt",
        _string(prompt),
        "--num-steps",
        str(int(num_steps)),
        "--guidance",
        str(float(guidance)),
        "--seed",
        str(int(seed)),
        "--num-frames",
        str(max(1, int(num_frames))),
        "--height",
        str(int(height)),
        "--width",
        str(int(width)),
        "--fps",
        str(float(fps)),
        "--output",
        str(output_video),
    ]
    if skeleton_video is not None:
        argv.extend(["--skeleton-video", str(skeleton_video)])
    return argv


def make_local_oscar_subprocess_generate(
    *,
    oscar_repo: str | Path,
    checkpoint: str | Path,
    python: str = "python",
    num_steps: int = 35,
    guidance: float = 6.0,
    height: int = 480,
    width: int = 640,
    fps: float = 15.0,
    run: Callable[..., Any],
    build_skeleton_video: Callable[[Sequence[Mapping[str, Any]], Path], Path | None] | None = None,
    extract_next_frame: Callable[[Path, Path], Path | None],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Real per-step OSCAR-2B inference, for running ON a GPU pod that has the OSCAR repo +
    checkpoint. ``run`` (subprocess.run), ``build_skeleton_video`` (landmarks -> conditioning
    video), and ``extract_next_frame`` (output clip -> next-observation frame, e.g. via ffmpeg)
    are injected, so the whole wrapper is unit-testable without GPU or OSCAR installed.
    """
    repo = Path(oscar_repo).expanduser()

    def _oscar_generate(request: Mapping[str, Any]) -> dict[str, Any]:
        out_dir = Path(_string(request.get("output_dir"))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_dir / "oscar_next_observation.mp4"
        landmarks = request.get("skeleton_landmarks") or []
        skeleton_video = (
            build_skeleton_video(landmarks, out_dir) if (build_skeleton_video and landmarks) else None
        )
        argv = build_oscar_inference_argv(
            python=python,
            oscar_repo=repo,
            checkpoint=checkpoint,
            first_frame_path=_string(request.get("reference_frame_path")),
            prompt=_string(request.get("task_prompt")),
            num_frames=int(request.get("num_frames") or 8),
            num_steps=num_steps,
            guidance=guidance,
            seed=int(request.get("seed") or 42),
            height=height,
            width=width,
            fps=fps,
            output_video=output_video,
            skeleton_video=skeleton_video,
        )
        completed = run(argv, cwd=str(repo), capture_output=True, text=True, check=False)
        returncode = getattr(completed, "returncode", 1)
        if returncode != 0 or not output_video.is_file():
            return {
                "status": "blocked",
                "blockers": [f"oscar_per_step_inference_returncode_{returncode}"],
                "generated_frame_path": "",
                "generated_video_path": str(output_video) if output_video.is_file() else "",
            }
        next_frame = extract_next_frame(output_video, out_dir)
        if not next_frame or not Path(next_frame).is_file():
            return {
                "status": "blocked",
                "blockers": ["oscar_per_step_next_frame_extraction_failed"],
                "generated_frame_path": "",
                "generated_video_path": str(output_video),
            }
        return {
            "status": "completed",
            "generated_frame_path": str(next_frame),
            "generated_video_path": str(output_video),
        }

    return _oscar_generate


def run_oscar_isaac_closed_loop(
    *,
    output_dir: str | Path,
    start_frame_path: str | Path,
    route_points: Sequence[Sequence[float]],
    wam_generate_next: WamGenerateNext,
    steps: int,
    probe_collision: Callable[[Sequence[float], float], int] | None = None,
    harness_backend_kind: str = "fixture",
    harness_backend_command: str | Sequence[str] | None = None,
    allow_external_backend: bool = False,
    backend_timeout_seconds: int = 600,
    policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    resolved_out = Path(output_dir).expanduser().resolve()
    ensure_dir(resolved_out)
    harness_dir = resolved_out / "wam_derived_observation_harness"
    route = [tuple(float(c) for c in point) for point in route_points]
    if not route:
        return {
            "schema_version": LOOP_SCHEMA_VERSION,
            "generated_at": generated,
            "status": "blocked",
            "blockers": ["blocked_empty_route"],
        }
    target = route[-1]
    bounded_steps = max(1, int(steps))

    policy = DeterministicWalkToTargetPolicy()
    policy.reset({"route_points": list(route), "start": route[0], "target": target})
    oracle = probe_collision or (lambda pose, yaw: 0)

    current_frame = str(Path(start_frame_path).expanduser().resolve())
    action_history: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    for step_index in range(1, bounded_steps + 1):
        # 1. policy acts
        decision = policy.step(
            StepContext(step=step_index - 1, num_steps=bounded_steps, probe_collision=oracle)
        )
        sim_time_s = round((step_index - 1) * 0.02, 9)
        action = action_record(
            decision=decision, step=step_index - 1, sim_time_s=sim_time_s, target=target
        )
        action_history.append(action)

        # 2. WAM generates the NEXT observation conditioned on the action
        wam_output = dict(
            wam_generate_next(current_frame, action, step_index, list(action_history)) or {}
        )
        generated_frame = _string(wam_output.get("generated_frame_path"))
        if not generated_frame or not Path(generated_frame).is_file():
            blockers.append(f"blocked_wam_generation_missing_frame_at_step_{step_index}")
            break

        # 3. perception harness (SAM3/DA3) analyses the generated frame immediately
        result = run_wam_derived_observation_harness_step(
            output_dir=harness_dir,
            generated_at=generated,
            step_index=step_index,
            source_generated_frame_path=generated_frame,
            source_generated_video_path=wam_output.get("generated_video_path"),
            source_wam_rollout_id=f"oscar_isaac_closed_loop_step_{step_index:04d}",
            transition_id=f"oscar_isaac_transition_{step_index:04d}",
            source_policy_action=action,
            action_history=action_history,
            current_policy_observation=_policy_observation(current_frame, target, step_index),
            skeleton_conditioning=wam_output.get("skeleton_conditioning"),
            previous_steps=step_records,
            previous_adapter_reports=adapter_reports,
            backend_kind=harness_backend_kind,
            backend_command=harness_backend_command,
            allow_external_backend=allow_external_backend,
            backend_timeout_seconds=backend_timeout_seconds,
            policy_id=policy_id,
        )
        step_record = dict(result.get("step_record") or {})
        adapter_report = dict(result.get("policy_adapter_report") or {})
        step_records.append(step_record)
        adapter_reports.append(adapter_report)
        trace_rows.append(
            {
                "step_index": step_index,
                "policy_action": action.get("policy_action"),
                "root_position": action.get("root_position"),
                "source_observation_frame": current_frame,
                "wam_generated_frame": generated_frame,
                "harness_step_status": step_record.get("status"),
                "harness_backend_kind": harness_backend_kind,
            }
        )

        # 4. feed forward: the generated frame becomes the next step's observation
        current_frame = generated_frame

    trace_path = resolved_out / "oscar_isaac_closed_loop_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    final_pose = trace_rows[-1]["root_position"] if trace_rows else list(route[0])
    reached = bool(
        trace_rows
        and interpolate_route(route, 1.0)[0]
        and sum((a - b) ** 2 for a, b in zip(final_pose, target)) ** 0.5 < 0.25
    )
    status = "completed" if trace_rows and not blockers else "blocked"
    manifest = {
        "schema_version": LOOP_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "loop_kind": "per_step_policy_wam_perception_closed_loop",
        "steps_executed": len(trace_rows),
        "steps_requested": bounded_steps,
        "harness_backend_kind": harness_backend_kind,
        "real_perception_backend_used": harness_backend_kind != "fixture",
        "task_target_position_xyz": [round(float(c), 6) for c in target],
        "final_root_position": final_pose,
        "task_target_reached": reached,
        "trace_path": str(trace_path),
        "harness_dir": str(harness_dir),
        "blockers": blockers,
        "claim_boundary": (
            "Per-step closed loop: policy action -> WAM-generated next observation -> SAM3/DA3 "
            "perception harness, repeated. Harness derives support observations from WAM pixels; "
            "task success still requires an external judge, not harness output alone."
        ),
        "raw_secret_values_recorded": False,
    }
    write_json(resolved_out / "oscar_isaac_closed_loop_manifest.json", manifest)
    return manifest


DEFAULT_SAM3_HARNESS_BACKEND_COMMAND = [
    sys.executable,
    "-m",
    "blueprint_pipeline.wam_real_provider_validation_probe",
    "backend",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the per-step OSCAR-2B <-> SAM3 closed loop. Intended to run ON a GPU pod that has the
    OSCAR repo + checkpoint and the SAM3/DA3 perception backend. ``--dry-run`` validates the full
    assembly (paths, backends, route) and writes the plan without any inference, so the wiring is
    verifiable with zero GPU.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-frame", required=True, help="initial robot-POV observation frame")
    parser.add_argument("--route-file", required=True, help='JSON: {"route_points": [[x,y,z],...]}')
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--task-prompt", default="walk to the sink")
    parser.add_argument("--num-frames", type=int, default=8, help="OSCAR clip length per step")
    parser.add_argument("--oscar-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--harness-backend-kind", default="real_provider_probe")
    parser.add_argument("--harness-backend-command", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)
    route = list(json.loads(Path(args.route_file).read_text(encoding="utf-8")).get("route_points") or [])
    harness_command = (
        shlex.split(args.harness_backend_command)
        if args.harness_backend_command
        else DEFAULT_SAM3_HARNESS_BACKEND_COMMAND
    )
    oscar_ready = bool(args.oscar_repo and args.checkpoint)

    if args.dry_run or not oscar_ready:
        plan = {
            "schema_version": "oscar_isaac_closed_loop_plan.v1",
            "generated_at": utc_now_iso(),
            "status": "prepared" if oscar_ready else "blocked",
            "mode": "dry_run" if args.dry_run else "prepared",
            "start_frame": args.start_frame,
            "start_frame_present": Path(args.start_frame).expanduser().is_file(),
            "route_point_count": len(route),
            "steps": int(args.steps),
            "task_prompt": args.task_prompt,
            "num_frames_per_step": int(args.num_frames),
            "oscar_repo": args.oscar_repo,
            "checkpoint_configured": bool(args.checkpoint),
            "harness_backend_kind": args.harness_backend_kind,
            "harness_backend_command_argv0": harness_command[0] if harness_command else None,
            "blockers": [] if oscar_ready else ["blocked_missing_oscar_repo_or_checkpoint"],
        }
        write_json(out_dir / "oscar_isaac_closed_loop_plan.json", plan)
        print(json.dumps({"status": plan["status"], "mode": plan["mode"]}, sort_keys=True))
        return 0 if plan["status"] in {"prepared"} else 2

    import subprocess

    oscar_generate = make_local_oscar_subprocess_generate(
        oscar_repo=args.oscar_repo,
        checkpoint=args.checkpoint,
        run=subprocess.run,
        extract_next_frame=extract_last_frame_via_opencv,
    )
    backend = make_oscar_per_step_wam_backend(
        oscar_generate=oscar_generate,
        work_dir=out_dir / "oscar_generation",
        task_prompt=args.task_prompt,
        num_frames=int(args.num_frames),
    )
    manifest = run_oscar_isaac_closed_loop(
        output_dir=out_dir,
        start_frame_path=args.start_frame,
        route_points=route,
        wam_generate_next=backend,
        steps=int(args.steps),
        harness_backend_kind=args.harness_backend_kind,
        harness_backend_command=harness_command,
        allow_external_backend=args.harness_backend_kind != "fixture",
    )
    print(json.dumps({"status": manifest["status"], "steps_executed": manifest.get("steps_executed")}, sort_keys=True))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
