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

import json
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
