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
import os
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .isaac_g1_policy import (
    DeterministicWalkToTargetPolicy,
    StepContext,
    action_record,
    interpolate_route,
)
from .oscar_wam_provider_command_adapter import run as run_oscar_wam_provider_adapter
from .wam_backend_strategy import get_wam_backend_strategy
from .wam_derived_observation_harness import run_wam_derived_observation_harness_step
from .wam_provider_runtime import WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE

LOOP_SCHEMA_VERSION = "oscar_isaac_closed_loop_eval.v1"
NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION = "oscar_next_observation_selection.v1"
CLOSED_LOOP_WAM_BACKEND_READINESS_SCHEMA_VERSION = "closed_loop_wam_backend_readiness.v1"
SUPPORTED_CLOSED_LOOP_WAM_BACKENDS = ("oscar_wam", "cosmos3_wam")
BUILT_IN_CLOSED_LOOP_WAM_BACKENDS = frozenset({"oscar_wam"})
VAST_API_GATE_ENV = "BLUEPRINT_ALLOW_VAST_API_CALLS"
VAST_INSTANCE_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"
VAST_PAID_WAM_GATE_ENV = "BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH"
VAST_API_KEY_FILE_ENV = "VAST_API_KEY_FILE"

# A WAM generation backend: given the current observation frame, the policy action, the step
# index, and the action history, produce the next-observation frame path (and optional video).
# Returns a mapping with at least {"generated_frame_path": <path>}.
WamGenerateNext = Callable[
    [str, Mapping[str, Any], int, Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    NEXT observation is selected from usable future frames of that clip. This is pure request
    shaping with no GPU or OSCAR import, so it is fully unit-testable; the actual inference is the
    injected callable in :func:`make_oscar_per_step_wam_backend`.
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


def build_wam_generation_step_input(
    *,
    current_frame_path: str | Path,
    action: Mapping[str, Any],
    step_index: int,
    output_dir: str | Path,
    task_prompt: str,
    next_observation_frame_path: str | Path | None = None,
    target_object_id: str = "task_target",
    projected_skeleton_trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the provider-bundle input for one per-step OSCAR WAM call."""
    frame = Path(current_frame_path).expanduser().resolve()
    visual = {
        "camera_id": "head_pov",
        "camera_frame_path": str(frame),
        "wam_generated_observation": step_index > 1,
    }
    if projected_skeleton_trace_path:
        trace_path = Path(projected_skeleton_trace_path).expanduser().resolve()
        visual["g1_projected_skeleton_trace_jsonl"] = str(trace_path)
        visual["projected_skeleton_trace_path"] = str(trace_path)
    out = Path(output_dir).expanduser().resolve()
    requested_next = (
        Path(next_observation_frame_path).expanduser()
        if next_observation_frame_path
        else out / "generated_next_observation.png"
    )
    return {
        "schema_version": "wam_generation_step_input.v1",
        "step_index": int(step_index),
        "source_policy_observation_frame_path": str(frame),
        "source_policy_action": {
            **dict(action),
            "task_prompt": _string(task_prompt),
            "action_type": _string(action.get("action_type"))
            or _string(action.get("policy_action"))
            or "isaac_g1_policy_action",
        },
        "current_policy_observation": {
            "schema_version": "blueprint_policy_observation.v1",
            "task_id": "isaac_g1_oscar_per_step_closed_loop",
            "target_object_id": target_object_id,
            "robot_profile_id": "unitree_g1",
            "policy_source": "isaac_g1_policy",
            "camera_frame_path": str(frame),
            "visual_observation": visual,
            "claim_boundary": {
                "simulator_generated_world_observation_only": True,
                "generated_wam_frame_is_support_artifact": step_index > 1,
                "physical_robot_sensor_proof": False,
                "deployment_readiness_proven": False,
            },
        },
        "requested_output": {
            "next_observation_frame_path": str(requested_next),
            "action_conditioned_generation_required": True,
        },
        "claim_boundary": {
            "isaac_policy_action_is_sim_policy_action": True,
            "wam_generation_is_not_robot_policy": True,
            "physical_robot_sensor_proof": False,
        },
    }


@contextmanager
def _temporary_environ(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _provider_video_path(payload: Mapping[str, Any]) -> str:
    for rollout in payload.get("rollouts", []) or []:
        if not isinstance(rollout, Mapping):
            continue
        video = _string(rollout.get("generated_video_path"))
        if video and Path(video).expanduser().is_file():
            return video
    return ""


def _provider_payload_proves_fresh_model(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("status") == "completed"
        and payload.get("fresh_provider_model_run_claimed")
        and payload.get("provider_learned_wam_model_ran")
        and payload.get("provider_generated_video_is_model_output")
    )


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string(os.getenv(name)) or default)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(_string(os.getenv(name)) or default))
    except ValueError:
        return default


def _vast_session_budget_path() -> Path:
    explicit = _string(os.getenv("VAST_SESSION_BUDGET_LEDGER_FILE"))
    if explicit:
        return Path(explicit).expanduser()
    key_file = Path(
        _string(os.getenv(VAST_API_KEY_FILE_ENV)) or "~/.blueprint-secrets/vast_api_key"
    ).expanduser()
    return key_file.parent / "vast_session_cost_summary.json"


def _vast_paid_provider_preflight(
    *,
    allow_paid_provider_launch: bool,
    max_hourly_rate_usd: float,
    max_live_minutes: int,
    session_max_live_minutes: int,
    hard_cap_usd: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not allow_paid_provider_launch:
        blockers.append("closed_loop_paid_provider_launch_not_authorized")
    if not _env_truthy(VAST_PAID_WAM_GATE_ENV):
        blockers.append(f"missing_env_{VAST_PAID_WAM_GATE_ENV}")
    if not _env_truthy(VAST_API_GATE_ENV):
        blockers.append(f"missing_env_{VAST_API_GATE_ENV}")
    if not _env_truthy(VAST_INSTANCE_LAUNCH_GATE_ENV):
        blockers.append(f"missing_env_{VAST_INSTANCE_LAUNCH_GATE_ENV}")
    key_file = Path(
        _string(os.getenv(VAST_API_KEY_FILE_ENV)) or "~/.blueprint-secrets/vast_api_key"
    ).expanduser()
    if not key_file.is_file():
        blockers.append(f"missing_file_based_secret_{VAST_API_KEY_FILE_ENV}")
    budget_path = _vast_session_budget_path()
    prior_cost = 0.0
    prior_live_seconds = 0.0
    budget_present = budget_path.is_file()
    budget_parse_error = None
    attempt_count = 0
    if budget_present:
        try:
            budget = json.loads(budget_path.read_text(encoding="utf-8"))
            attempts = budget.get("attempts")
            if isinstance(attempts, list):
                attempt_count = len(attempts)
                for row in attempts:
                    if not isinstance(row, Mapping):
                        continue
                    try:
                        prior_cost += float(row.get("estimated_cost_usd") or 0.0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        prior_live_seconds += float(
                            row.get("actual_live_runtime_seconds_observed_by_adapter")
                            or 0.0
                        )
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:  # pragma: no cover - type surfaced in artifact
            budget_parse_error = type(exc).__name__
            blockers.append("vast_session_budget_ledger_parse_failed")
    projected_incremental_cost = max_hourly_rate_usd * (max_live_minutes / 60.0)
    prior_live_minutes = prior_live_seconds / 60.0
    if (
        session_max_live_minutes >= 0
        and prior_live_minutes >= float(session_max_live_minutes)
    ):
        blockers.append("session_live_runtime_limit_exhausted")
    if prior_cost + projected_incremental_cost > hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_exhausted")
    elif prior_cost >= hard_cap_usd:
        blockers.append("session_estimated_spend_hard_cap_already_exceeded")
    if prior_cost > 0.0 and not budget_present:
        warnings.append("vast_budget_prior_cost_inferred_without_ledger")
    return {
        "schema_version": "closed_loop_vast_paid_provider_preflight.v1",
        "status": "ready" if not blockers else "blocked",
        "provider": "vast",
        "gate_env": {
            VAST_PAID_WAM_GATE_ENV: _env_truthy(VAST_PAID_WAM_GATE_ENV),
            VAST_API_GATE_ENV: _env_truthy(VAST_API_GATE_ENV),
            VAST_INSTANCE_LAUNCH_GATE_ENV: _env_truthy(VAST_INSTANCE_LAUNCH_GATE_ENV),
        },
        "vast_api_key_file_present": key_file.is_file(),
        "vast_api_key_file_path": str(key_file),
        "budget_path": str(budget_path),
        "budget_ledger_present": budget_present,
        "budget_parse_error": budget_parse_error,
        "attempt_count": attempt_count,
        "prior_estimated_cost_usd": round(prior_cost, 6),
        "prior_live_runtime_minutes": round(prior_live_minutes, 6),
        "max_hourly_rate_usd": float(max_hourly_rate_usd),
        "requested_max_live_runtime_minutes": int(max_live_minutes),
        "session_max_live_runtime_minutes": int(session_max_live_minutes),
        "projected_max_incremental_cost_usd": round(projected_incremental_cost, 6),
        "hard_cap_usd": float(hard_cap_usd),
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "raw_secret_values_recorded": False,
        "claim_boundary": {
            "preflight_is_no_spend": True,
            "preflight_does_not_call_vast_api": True,
            "secret_values_not_read_into_artifact": True,
        },
    }


def _closed_loop_paid_provider_preflight(
    *,
    provider: str,
    allow_paid_provider_launch: bool,
) -> dict[str, Any]:
    provider_id = _string(provider).strip().lower()
    if provider_id != "vast":
        return {
            "schema_version": "closed_loop_paid_provider_preflight.v1",
            "status": "not_applicable",
            "provider": provider_id or provider,
            "blockers": [],
            "claim_boundary": {"preflight_is_no_spend": True},
        }
    return _vast_paid_provider_preflight(
        allow_paid_provider_launch=allow_paid_provider_launch,
        max_hourly_rate_usd=_float_env("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", 0.35),
        max_live_minutes=_int_env("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", 30),
        session_max_live_minutes=_int_env("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", 35),
        hard_cap_usd=_float_env("BLUEPRINT_VAST_WAM_HARD_CAP_USD", 3.0),
    )


def build_closed_loop_wam_backend_readiness(
    *,
    selected_backend: str,
    use_provider_command: bool,
    oscar_repo: str | None = None,
    checkpoint: str | None = None,
    oscar_provider: str = "runpod",
    allow_paid_provider_launch: bool = False,
) -> dict[str, Any]:
    """Describe which WAM backend the closed-loop runner can actually execute.

    The strategy catalog can prefer Cosmos3 for new learned-WAM work, but this
    runner still has only OSCAR-specific local/provider execution paths. This
    manifest is a no-spend guardrail so a paid run cannot be mistaken for a
    Cosmos3 run unless a real Cosmos3 adapter has been wired through the loop.
    """

    backend = _string(selected_backend).strip() or "oscar_wam"
    command_env_var = WAM_PROVIDER_COMMAND_ENV_BY_SUBSTRATE.get(backend)
    backend_command = (
        _string(os.environ.get(command_env_var or ""))
        or _string(os.environ.get("BLUEPRINT_WAM_PROVIDER_COMMAND"))
    )
    local_oscar_configured = bool(_string(oscar_repo) and _string(checkpoint))
    built_in_oscar_provider_configured = bool(
        backend == "oscar_wam" and use_provider_command
    )
    paid_provider_preflight = (
        _closed_loop_paid_provider_preflight(
            provider=oscar_provider,
            allow_paid_provider_launch=allow_paid_provider_launch,
        )
        if built_in_oscar_provider_configured and allow_paid_provider_launch
        else {
            "schema_version": "closed_loop_paid_provider_preflight.v1",
            "status": "not_requested",
            "provider": oscar_provider,
            "blockers": [],
            "claim_boundary": {"preflight_is_no_spend": True},
        }
    )
    explicit_provider_command_configured = bool(backend_command)
    supported_by_this_runner = backend in BUILT_IN_CLOSED_LOOP_WAM_BACKENDS
    blockers: list[str] = []
    if backend not in SUPPORTED_CLOSED_LOOP_WAM_BACKENDS:
        blockers.append("unsupported_closed_loop_wam_backend")
    if backend == "oscar_wam":
        if not (built_in_oscar_provider_configured or local_oscar_configured):
            blockers.append("blocked_missing_oscar_provider_or_local_checkpoint")
        blockers.extend(str(item) for item in paid_provider_preflight.get("blockers") or [])
    elif backend == "cosmos3_wam":
        blockers.append("blocked_cosmos3_wam_not_wired_into_isaac_closed_loop_runner")
        if not explicit_provider_command_configured:
            blockers.append("blocked_cosmos3_wam_requires_explicit_provider_command")
    elif backend in SUPPORTED_CLOSED_LOOP_WAM_BACKENDS:
        blockers.append("blocked_selected_wam_backend_not_supported_by_runner")
    return {
        "schema_version": CLOSED_LOOP_WAM_BACKEND_READINESS_SCHEMA_VERSION,
        "selected_wam_backend": backend,
        "status": "ready" if not blockers else "blocked",
        "supported_backend_ids": list(SUPPORTED_CLOSED_LOOP_WAM_BACKENDS),
        "built_in_closed_loop_backend_ids": sorted(BUILT_IN_CLOSED_LOOP_WAM_BACKENDS),
        "supported_by_this_runner": supported_by_this_runner,
        "provider_adapter_kind": (
            "built_in_oscar_provider_adapter"
            if built_in_oscar_provider_configured
            else "local_oscar_subprocess"
            if local_oscar_configured
            else "explicit_provider_command"
            if explicit_provider_command_configured
            else "not_configured"
        ),
        "oscar_provider": oscar_provider,
        "allow_paid_provider_launch": bool(allow_paid_provider_launch),
        "local_oscar_repo_configured": bool(_string(oscar_repo)),
        "local_oscar_checkpoint_configured": bool(_string(checkpoint)),
        "explicit_provider_command_configured": explicit_provider_command_configured,
        "provider_command_env_var": command_env_var,
        "generic_provider_command_env_var": "BLUEPRINT_WAM_PROVIDER_COMMAND",
        "paid_provider_preflight": paid_provider_preflight,
        "strategy": get_wam_backend_strategy(backend),
        "blockers": blockers,
        "claim_boundary": {
            "readiness_manifest_is_no_spend": True,
            "readiness_manifest_is_not_model_execution_proof": True,
            "cosmos3_strategy_preference_does_not_imply_runtime_wired": True,
            "oscar_provider_path_is_not_cosmos3_runtime": backend == "oscar_wam",
        },
    }


def make_oscar_provider_command_wam_backend(
    *,
    work_dir: str | Path,
    task_prompt: str,
    num_frames: int = 8,
    num_steps: int = 35,
    guidance: float = 6.0,
    seed: int = 42,
    height: int = 480,
    width: int = 640,
    fps: float = 15.0,
    provider: str = "runpod",
    allow_paid_provider_launch: bool = False,
    timeout_seconds: float = 3600.0,
    adapter_run: Callable[[Sequence[str] | None], Mapping[str, Any]] = run_oscar_wam_provider_adapter,
    extract_next_frame: Callable[[str | Path, str | Path], Path | None] | None = None,
    projected_skeleton_trace_path: str | Path | None = None,
) -> WamGenerateNext:
    """Drive one fresh OSCAR provider run per closed-loop step."""
    resolved_work = Path(work_dir).expanduser().resolve()
    ensure_dir(resolved_work)

    def _generate_next(
        current_frame: str,
        action: Mapping[str, Any],
        step_index: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        step_dir = resolved_work / f"step_{step_index:04d}"
        ensure_dir(step_dir)
        step_input = build_wam_generation_step_input(
            current_frame_path=current_frame,
            action=action,
            step_index=step_index,
            output_dir=step_dir,
            task_prompt=task_prompt,
            projected_skeleton_trace_path=projected_skeleton_trace_path if step_index == 1 else None,
        )
        step_input_path = step_dir / "wam_generation_step_input.json"
        write_json(step_input_path, step_input)
        output_path = step_dir / "wam_provider_output.json"
        adapter_args = [
            "--mode",
            "auto",
            "--provider",
            provider,
            "--work-dir",
            str(step_dir / "provider_workspace"),
            "--timeout-seconds",
            str(float(timeout_seconds)),
        ]
        if allow_paid_provider_launch:
            adapter_args.append("--allow-paid-provider-launch")
        with _temporary_environ(
            {
                "BLUEPRINT_WAM_ROLLOUT_INPUT": str(step_input_path),
                "BLUEPRINT_WAM_ROLLOUT_OUTPUT": str(output_path),
                "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": str(max(1, int(num_frames))),
                "BLUEPRINT_OSCAR_WAM_NUM_STEPS": str(max(1, int(num_steps))),
                "BLUEPRINT_OSCAR_WAM_GUIDANCE": str(float(guidance)),
                "BLUEPRINT_OSCAR_WAM_SEED": str(int(seed) + int(step_index)),
                "BLUEPRINT_OSCAR_WAM_HEIGHT": str(int(height)),
                "BLUEPRINT_OSCAR_WAM_WIDTH": str(int(width)),
                "BLUEPRINT_OSCAR_WAM_FPS": str(float(fps)),
            }
        ):
            payload = dict(adapter_run(adapter_args) or {})
        if not output_path.is_file():
            write_json(output_path, payload)
        video = _provider_video_path(payload)
        if not video:
            return {
                "status": "blocked",
                "wam_backend": "oscar_2b_per_step_provider",
                "generated_frame_path": "",
                "generated_video_path": "",
                "provider_payload": payload,
                "provider_output_path": str(output_path),
                "fresh_provider_model_run_claimed": False,
                "blockers": payload.get("blockers") or ["oscar_provider_video_missing"],
            }
        extractor = extract_next_frame or extract_next_observation_frame_from_video
        next_frame = extractor(video, step_dir / "next_observation")
        if next_frame is None or not Path(next_frame).is_file():
            return {
                "status": "blocked",
                "wam_backend": "oscar_2b_per_step_provider",
                "generated_frame_path": "",
                "generated_video_path": video,
                "provider_payload": payload,
                "provider_output_path": str(output_path),
                "fresh_provider_model_run_claimed": _provider_payload_proves_fresh_model(payload),
                "blockers": ["oscar_provider_next_observation_frame_extraction_failed"],
            }
        return {
            "status": "completed" if payload.get("status") == "completed" else "blocked",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(next_frame),
            "generated_video_path": video,
            "provider_payload": payload,
            "provider_output_path": str(output_path),
            "fresh_provider_model_run_claimed": _provider_payload_proves_fresh_model(payload),
            "blockers": payload.get("blockers") or [],
        }

    return _generate_next


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


def _provider_completed(provider_statuses: Sequence[Any], provider: str) -> bool:
    for status_value in provider_statuses:
        if not isinstance(status_value, Mapping):
            continue
        if status_value.get("provider") != provider:
            continue
        return bool(status_value.get("ran")) and not bool(status_value.get("blockers") or [])
    return False


def _da3_completed(provider_statuses: Sequence[Any]) -> bool:
    for status_value in provider_statuses:
        if not isinstance(status_value, Mapping):
            continue
        if status_value.get("provider") != "depth":
            continue
        kind = _string(status_value.get("kind")).lower()
        return bool(
            status_value.get("ran")
            and kind in {"depth_anything_3", "da3", "depth-anything-3"}
            and not bool(status_value.get("blockers") or [])
        )
    return False


def _step_backend_status(step_record: Mapping[str, Any]) -> dict[str, Any]:
    backend = step_record.get("harness_backend")
    if not isinstance(backend, Mapping):
        backend = step_record.get("backend") if isinstance(step_record.get("backend"), Mapping) else {}
    provider_statuses = list(backend.get("provider_statuses") or []) if isinstance(backend, Mapping) else []
    return {
        "backend_status": backend.get("status") if isinstance(backend, Mapping) else None,
        "real_model_ran": bool(
            isinstance(backend, Mapping) and backend.get("real_sam_or_depth_model_ran")
        ),
        "provider_statuses": provider_statuses,
        "sam3_completed": _provider_completed(provider_statuses, "sam3"),
        "depth_completed": _provider_completed(provider_statuses, "depth"),
        "da3_completed": _da3_completed(provider_statuses),
    }


def _frame_signal_stats(frame: Any, cv2: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    edges = cv2.Canny(gray, 50, 150)
    return {
        "mean_luma": round(float(gray.mean()), 3),
        "std_luma": round(float(gray.std()), 3),
        "luma_min": int(gray.min()),
        "luma_max": int(gray.max()),
        "luma_range": int(gray.max()) - int(gray.min()),
        "dark_pixel_ratio": round(float((gray < 32).mean()), 6),
        "bright_pixel_ratio": round(float((gray > 224).mean()), 6),
        "edge_density": round(float((edges > 0).mean()), 6),
    }


def _next_observation_signal_blockers(stats: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    mean_luma = float(stats.get("mean_luma") or 0.0)
    std_luma = float(stats.get("std_luma") or 0.0)
    luma_range = float(stats.get("luma_range") or 0.0)
    dark_ratio = float(stats.get("dark_pixel_ratio") or 0.0)
    bright_ratio = float(stats.get("bright_pixel_ratio") or 0.0)
    edge_density = float(stats.get("edge_density") or 0.0)
    if mean_luma < 25.0 or dark_ratio > 0.78:
        blockers.append("next_observation_candidate_too_dark")
    if mean_luma > 245.0 and bright_ratio > 0.90:
        blockers.append("next_observation_candidate_overexposed")
    if std_luma < 8.0 or luma_range < 32.0:
        blockers.append("next_observation_candidate_flat_or_low_contrast")
    if edge_density < 0.002:
        blockers.append("next_observation_candidate_low_scene_structure")
    if edge_density > 0.12 and std_luma < 28.0:
        blockers.append("next_observation_candidate_static_noise_artifact")
    return blockers


def _write_selection_manifest(
    out_dir: Path,
    *,
    status: str,
    video_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    selected_frame_index: int | None,
    blockers: Sequence[str],
    extraction_method: str,
) -> None:
    write_json(
        out_dir / "next_observation_selection.json",
        {
            "schema_version": NEXT_OBSERVATION_SELECTION_SCHEMA_VERSION,
            "status": status,
            "video_path": str(video_path),
            "selected_frame_index": selected_frame_index,
            "extraction_method": extraction_method,
            "candidate_count": len(candidates),
            "candidates": list(candidates),
            "blockers": list(blockers),
            "claim_boundary": {
                "selected_frame_is_generated_next_observation_candidate": status == "completed",
                "visual_signal_gate_is_not_task_success_evidence": True,
                "scene_or_task_specific_pixels_used": False,
            },
        },
    )


def extract_next_observation_frame_from_video(video_path: str | Path, out_dir: str | Path) -> Path | None:
    """Default ``extract_next_frame`` for OSCAR clips.

    The first video frame is treated as the seed/current observation. The next observation is the
    earliest future frame with enough generic visual signal to feed the harness. This avoids
    advancing the closed loop with late frames that have collapsed to dark/flat artifacts while
    keeping the gate task- and scene-neutral.
    """
    resolved_out = Path(out_dir).expanduser()
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_video = Path(video_path).expanduser()
    try:
        import cv2  # local import: only needed where a real clip is produced
    except ImportError:
        cv2 = None
    if cv2 is not None:
        capture = cv2.VideoCapture(str(resolved_video))
        candidates: list[dict[str, Any]] = []
        selected_index: int | None = None
        selected_frame = None
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                stats = _frame_signal_stats(frame, cv2)
                blockers = (
                    ["next_observation_candidate_is_seed_frame"]
                    if frame_index == 0
                    else _next_observation_signal_blockers(stats)
                )
                candidates.append(
                    {
                        "frame_index": frame_index,
                        **stats,
                        "blockers": blockers,
                    }
                )
                if frame_index > 0 and not blockers:
                    selected_index = frame_index
                    selected_frame = frame.copy()
                    break
                frame_index += 1
        finally:
            capture.release()
        if selected_frame is None:
            _write_selection_manifest(
                resolved_out,
                status="blocked",
                video_path=resolved_video,
                candidates=candidates,
                selected_frame_index=None,
                blockers=["no_usable_future_next_observation_frame"],
                extraction_method="opencv_signal_gate",
            )
            return None
        frame_path = resolved_out / "next_observation.png"
        if not cv2.imwrite(str(frame_path), selected_frame):
            _write_selection_manifest(
                resolved_out,
                status="blocked",
                video_path=resolved_video,
                candidates=candidates,
                selected_frame_index=selected_index,
                blockers=["next_observation_frame_write_failed"],
                extraction_method="opencv_signal_gate",
            )
            return None
        _write_selection_manifest(
            resolved_out,
            status="completed",
            video_path=resolved_video,
            candidates=candidates,
            selected_frame_index=selected_index,
            blockers=[],
            extraction_method="opencv_signal_gate",
        )
        return frame_path

    frame_path = resolved_out / "next_observation.png"
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(resolved_video),
                "-vf",
                "select=gte(n\\,1)",
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not frame_path.is_file():
        _write_selection_manifest(
            resolved_out,
            status="blocked",
            video_path=resolved_video,
            candidates=[],
            selected_frame_index=None,
            blockers=["ffmpeg_first_future_frame_extraction_failed"],
            extraction_method="ffmpeg_first_future_frame",
        )
        return None
    _write_selection_manifest(
        resolved_out,
        status="completed",
        video_path=resolved_video,
        candidates=[],
        selected_frame_index=1,
        blockers=[],
        extraction_method="ffmpeg_first_future_frame",
    )
    return frame_path


def extract_last_frame_via_opencv(video_path: str | Path, out_dir: str | Path) -> Path | None:
    """Compatibility wrapper for older callers.

    Despite the historical name, the closed-loop now extracts the earliest usable future frame
    rather than blindly taking the last frame.
    """
    return extract_next_observation_frame_from_video(video_path, out_dir)


def _geometry_sidecar_from_route(
    route_payload: Mapping[str, Any],
    *,
    start_frame_path: str | Path,
) -> Path | None:
    candidates: list[Any] = [
        route_payload.get("manipulation_pov_geometry_path"),
        route_payload.get("seed_geometry_path"),
        route_payload.get("geometry_path"),
    ]
    source_trace = _string(route_payload.get("source_trace"))
    if source_trace:
        candidates.append(Path(source_trace).expanduser().parent / "manipulation_pov_geometry.json")
    start = Path(start_frame_path).expanduser()
    candidates.append(start.parent / "manipulation_pov_geometry.json")
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    return None


def _infer_skeleton_segments(landmarks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    ids = {_string(item.get("landmark_id")) for item in landmarks}
    segments: list[dict[str, str]] = []
    for prefix in ("left", "right"):
        wrist = f"{prefix}_wrist_link"
        hand = f"{prefix}_hand_link"
        if wrist in ids and hand in ids:
            segments.append({"from": wrist, "to": hand})
    return segments


def _scaled_projection_xy(
    projection: Mapping[str, Any],
    *,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float] | None:
    if projection.get("available") is not True:
        return None
    try:
        return float(projection.get("u_px")) * x_scale, float(projection.get("v_px")) * y_scale
    except (TypeError, ValueError):
        return None


def _landmark_temporal_reach_fraction(landmark: Mapping[str, Any]) -> float:
    text = f"{_string(landmark.get('landmark_id'))} {_string(landmark.get('link_role'))}".lower()
    if "hand" in text or "gripper" in text:
        return 0.70
    if "wrist" in text:
        return 0.45
    if "elbow" in text or "forearm" in text:
        return 0.25
    if "shoulder" in text:
        return 0.08
    return 0.35


def _temporal_projected_landmarks(
    landmarks: Sequence[Mapping[str, Any]],
    *,
    target_projection_xy: tuple[float, float] | None,
    progress: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bounded_progress = max(0.0, min(1.0, float(progress)))
    for landmark in landmarks:
        item = dict(landmark)
        projection = dict(_mapping(item.get("image_projection")))
        if target_projection_xy is not None and projection.get("available") is True:
            try:
                u_px = float(projection.get("u_px"))
                v_px = float(projection.get("v_px"))
            except (TypeError, ValueError):
                output.append(item)
                continue
            reach = _landmark_temporal_reach_fraction(item) * bounded_progress
            projection["u_px"] = round(u_px + (target_projection_xy[0] - u_px) * reach, 3)
            projection["v_px"] = round(v_px + (target_projection_xy[1] - v_px) * reach, 3)
        item["image_projection"] = projection
        output.append(item)
    return output


def materialize_projected_skeleton_trace_from_seed_geometry(
    *,
    route_payload: Mapping[str, Any],
    start_frame_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 8,
) -> Path | None:
    """Convert a seed-render geometry sidecar into OSCAR's projected-skeleton trace format.

    This uses route/seed metadata only. It does not know about kitchens, refrigerators, or fixed
    coordinates; if no geometry sidecar is present, the caller simply proceeds without the trace.
    """
    geometry_path = _geometry_sidecar_from_route(route_payload, start_frame_path=start_frame_path)
    if geometry_path is None:
        return None
    try:
        payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else [payload]
    if not frames:
        return None
    row = next((item for item in frames if isinstance(item, Mapping)), None)
    if row is None:
        return None
    raw_landmarks = row.get("projected_landmarks")
    if not isinstance(raw_landmarks, Sequence) or isinstance(raw_landmarks, (str, bytes)):
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(Path(start_frame_path).expanduser()) as image:
            target_width, target_height = image.size
    except Exception:
        return None
    seed_quality = row.get("seed_frame_quality") if isinstance(row.get("seed_frame_quality"), Mapping) else {}
    image_size = seed_quality.get("image_size_px") if isinstance(seed_quality.get("image_size_px"), Sequence) else []
    source_width = float(image_size[0]) if len(image_size) >= 2 else float(target_width)
    source_height = float(image_size[1]) if len(image_size) >= 2 else float(target_height)
    if source_width <= 0.0 or source_height <= 0.0:
        return None
    x_scale = float(target_width) / source_width
    y_scale = float(target_height) / source_height
    landmarks: list[dict[str, Any]] = []
    for landmark in raw_landmarks:
        if not isinstance(landmark, Mapping):
            continue
        projection = landmark.get("image_projection")
        if not isinstance(projection, Mapping) or projection.get("available") is not True:
            continue
        projected_xy = _scaled_projection_xy(projection, x_scale=x_scale, y_scale=y_scale)
        if projected_xy is None:
            continue
        landmarks.append(
            {
                "landmark_id": _string(landmark.get("landmark_id")),
                "link_role": _string(landmark.get("link_role")),
                "image_projection": {
                    "available": True,
                    "u_px": round(projected_xy[0], 3),
                    "v_px": round(projected_xy[1], 3),
                    "depth_m": projection.get("depth_m"),
                },
            }
        )
    if not landmarks:
        return None
    raw_segments = row.get("segments") if isinstance(row.get("segments"), Sequence) else []
    segments = [
        {"from": _string(segment.get("from")), "to": _string(segment.get("to"))}
        for segment in raw_segments
        if isinstance(segment, Mapping) and segment.get("from") and segment.get("to")
    ]
    if not segments:
        segments = _infer_skeleton_segments(landmarks)
    target_projection = row.get("target_projection")
    target_projection_xy = (
        _scaled_projection_xy(target_projection, x_scale=x_scale, y_scale=y_scale)
        if isinstance(target_projection, Mapping)
        else None
    )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "g1_projected_skeleton_trace.jsonl"
    frame_count = max(1, int(num_frames)) if target_projection_xy is not None else 1
    base_frame_index = int(row.get("frame_index") or row.get("step") or 0)
    lines: list[str] = []
    for trace_index in range(frame_count):
        progress = trace_index / max(frame_count - 1, 1)
        trace_landmarks = _temporal_projected_landmarks(
            landmarks,
            target_projection_xy=target_projection_xy,
            progress=progress,
        )
        trace_row = {
            "schema_version": "blueprint.g1.projected_upper_body_skeleton.v1",
            "status": "completed",
            "source_geometry_path": str(geometry_path),
            "frame_index": base_frame_index + trace_index,
            "temporal_progress": round(progress, 6),
            "camera": _string(row.get("camera")) or "head_pov",
            "image_size_px": [int(target_width), int(target_height)],
            "source_image_size_px": [int(source_width), int(source_height)],
            "target_projection": {
                "available": target_projection_xy is not None,
                "u_px": round(target_projection_xy[0], 3) if target_projection_xy else None,
                "v_px": round(target_projection_xy[1], 3) if target_projection_xy else None,
            },
            "projected_landmark_count": len(trace_landmarks),
            "landmarks": trace_landmarks,
            "segments": segments,
            "claim_boundary": {
                "projected_skeleton_trace_derived_from_seed_render_geometry": True,
                "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": bool(
                    target_projection_xy
                ),
                "not_a_learned_robot_policy_action": True,
                "simulated_state_not_physical_robot_sensor_evidence": True,
                "not_task_success_or_contact_proof": True,
            },
        }
        lines.append(json.dumps(trace_row, sort_keys=True))
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return trace_path


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
    require_fresh_oscar_provider: bool = False,
    require_real_perception_backend: bool = False,
    require_sam3_completed: bool = False,
    require_da3_completed: bool = False,
    perception_target_prompts: Sequence[str] | None = None,
    wam_backend_id: str = "oscar_wam",
    wam_backend_readiness: Mapping[str, Any] | None = None,
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
    cleaned_target_prompts = [
        prompt for prompt in (_string(item) for item in (perception_target_prompts or [])) if prompt
    ]
    bounded_steps = max(1, int(steps))

    policy = DeterministicWalkToTargetPolicy()
    policy.reset({"route_points": list(route), "start": route[0], "target": target})
    oracle = probe_collision or (lambda pose, yaw: 0)

    current_frame = str(Path(start_frame_path).expanduser().resolve())
    action_history: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    proof_rows: list[dict[str, Any]] = []
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
        wam_provider_payload = (
            wam_output.get("provider_payload")
            if isinstance(wam_output.get("provider_payload"), Mapping)
            else {}
        )
        fresh_oscar_provider = bool(
            wam_output.get("fresh_provider_model_run_claimed")
            or _provider_payload_proves_fresh_model(wam_provider_payload)
        )
        if require_fresh_oscar_provider and not fresh_oscar_provider:
            blockers.append(f"fresh_oscar_provider_model_run_not_proven_at_step_{step_index}")

        # 3. perception harness (SAM3/DA3) analyses the generated frame immediately
        result = run_wam_derived_observation_harness_step(
            output_dir=harness_dir,
            generated_at=generated,
            step_index=step_index,
            source_generated_frame_path=generated_frame,
            source_generated_video_path=wam_output.get("generated_video_path"),
            source_wam_rollout_id=f"oscar_isaac_closed_loop_step_{step_index:04d}",
            transition_id=f"oscar_isaac_transition_{step_index:04d}",
            source_policy_action={
                **action,
                **({"task_prompt": cleaned_target_prompts[0]} if cleaned_target_prompts else {}),
            },
            action_history=action_history,
            current_policy_observation=_policy_observation(current_frame, target, step_index),
            skeleton_conditioning=wam_output.get("skeleton_conditioning"),
            eval_ready_task_grounding={
                "schema_version": "eval_ready_task_grounding.v1",
                "status": "prompt_only_for_generated_frame_perception"
                if cleaned_target_prompts
                else "not_supplied",
                "task": {
                    "task_id": "isaac_g1_oscar_per_step_closed_loop",
                    "target_prompts_for_object_index_backends": cleaned_target_prompts,
                },
                "selected_task_target": {
                    "object_id": "perception_target",
                    "label": cleaned_target_prompts[0],
                    "source_prompt": cleaned_target_prompts[0],
                    "source": "closed_loop_cli_target_prompt",
                }
                if cleaned_target_prompts
                else {},
            },
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
        backend_status = _step_backend_status(step_record)
        if require_real_perception_backend and not backend_status["real_model_ran"]:
            blockers.append(f"real_perception_backend_not_proven_at_step_{step_index}")
        if require_sam3_completed and not backend_status["sam3_completed"]:
            blockers.append(f"sam3_provider_not_completed_at_step_{step_index}")
        if require_da3_completed and not backend_status["da3_completed"]:
            blockers.append(f"da3_provider_not_completed_at_step_{step_index}")
        step_records.append(step_record)
        adapter_reports.append(adapter_report)
        trace_row = {
            "step_index": step_index,
            "policy_action": action.get("policy_action"),
            "root_position": action.get("root_position"),
            "source_observation_frame": current_frame,
            "wam_generated_frame": generated_frame,
            "wam_generated_video": wam_output.get("generated_video_path"),
            "wam_backend": wam_output.get("wam_backend"),
            "wam_generation_status": wam_output.get("status") or wam_output.get("wam_generation_status"),
            "fresh_oscar_provider_model_run_claimed": fresh_oscar_provider,
            "provider_output_path": wam_output.get("provider_output_path"),
            "harness_step_status": step_record.get("status"),
            "harness_backend_kind": harness_backend_kind,
            "real_perception_backend_model_ran": backend_status["real_model_ran"],
            "sam3_completed": backend_status["sam3_completed"],
            "depth_completed": backend_status["depth_completed"],
            "da3_completed": backend_status["da3_completed"],
        }
        trace_rows.append(trace_row)
        proof_rows.append(
            {
                "step_index": step_index,
                "policy_action_recorded": bool(action.get("policy_action")),
                "source_observation_frame": current_frame,
                "wam_generated_frame": generated_frame,
                "wam_generated_video": wam_output.get("generated_video_path"),
                "oscar_per_step_backend": wam_output.get("wam_backend"),
                "fresh_oscar_provider_model_run_claimed": fresh_oscar_provider,
                "real_perception_backend_model_ran": backend_status["real_model_ran"],
                "sam3_completed": backend_status["sam3_completed"],
                "depth_completed": backend_status["depth_completed"],
                "da3_completed": backend_status["da3_completed"],
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
    feed_forward_verified = all(
        trace_rows[index]["source_observation_frame"]
        == trace_rows[index - 1]["wam_generated_frame"]
        for index in range(1, len(trace_rows))
    )
    proof = {
        "policy_source": "isaac_g1_policy.DeterministicWalkToTargetPolicy",
        "selected_wam_backend": _string(wam_backend_id) or "oscar_wam",
        "isaac_policy_actions_recorded": len(action_history),
        "oscar_per_step_generation_calls": sum(
            1 for row in proof_rows if row.get("oscar_per_step_backend")
        ),
        "fresh_oscar_provider_model_run_steps": sum(
            1 for row in proof_rows if row.get("fresh_oscar_provider_model_run_claimed")
        ),
        "real_perception_backend_steps": sum(
            1 for row in proof_rows if row.get("real_perception_backend_model_ran")
        ),
        "sam3_completed_steps": sum(1 for row in proof_rows if row.get("sam3_completed")),
        "depth_completed_steps": sum(1 for row in proof_rows if row.get("depth_completed")),
        "da3_completed_steps": sum(1 for row in proof_rows if row.get("da3_completed")),
        "feed_forward_verified": feed_forward_verified,
        "requirements": {
            "fresh_oscar_provider_required": bool(require_fresh_oscar_provider),
            "real_perception_backend_required": bool(require_real_perception_backend),
            "sam3_completed_required": bool(require_sam3_completed),
            "da3_completed_required": bool(require_da3_completed),
        },
        "per_step": proof_rows,
    }
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
        "perception_target_prompts": cleaned_target_prompts,
        "final_root_position": final_pose,
        "task_target_reached": reached,
        "trace_path": str(trace_path),
        "harness_dir": str(harness_dir),
        "selected_wam_backend": _string(wam_backend_id) or "oscar_wam",
        "wam_backend_readiness": dict(wam_backend_readiness or {}),
        "proof": proof,
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
    parser.add_argument("--oscar-num-steps", type=int, default=35)
    parser.add_argument("--oscar-guidance", type=float, default=6.0)
    parser.add_argument("--oscar-seed", type=int, default=42)
    parser.add_argument("--oscar-height", type=int, default=480)
    parser.add_argument("--oscar-width", type=int, default=640)
    parser.add_argument("--oscar-fps", type=float, default=15.0)
    parser.add_argument(
        "--wam-backend",
        choices=SUPPORTED_CLOSED_LOOP_WAM_BACKENDS,
        default="oscar_wam",
        help=(
            "WAM backend requested for the closed-loop. This runner currently "
            "has built-in execution only for oscar_wam; cosmos3_wam is a "
            "blocked readiness check until an explicit Cosmos3 adapter is wired."
        ),
    )
    parser.add_argument("--oscar-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--use-provider-command", action="store_true")
    parser.add_argument("--oscar-provider", choices=("auto", "vast", "runpod"), default="runpod")
    parser.add_argument("--provider-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--allow-paid-provider-launch", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--harness-backend-kind", default="real_provider_probe")
    parser.add_argument("--harness-backend-command", default=None)
    parser.add_argument("--perception-target-prompt", action="append", default=[])
    parser.add_argument("--require-fresh-oscar-provider", action="store_true")
    parser.add_argument("--require-real-perception-backend", action="store_true")
    parser.add_argument("--require-sam3-completed", action="store_true")
    parser.add_argument("--require-da3-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_dir)
    route_payload = json.loads(Path(args.route_file).read_text(encoding="utf-8"))
    route = list(route_payload.get("route_points") or [])
    projected_skeleton_trace_path = materialize_projected_skeleton_trace_from_seed_geometry(
        route_payload=route_payload,
        start_frame_path=args.start_frame,
        output_dir=out_dir / "seed_conditioning",
        num_frames=int(args.num_frames),
    )
    harness_command = (
        shlex.split(args.harness_backend_command)
        if args.harness_backend_command
        else DEFAULT_SAM3_HARNESS_BACKEND_COMMAND
    )
    wam_backend_readiness = build_closed_loop_wam_backend_readiness(
        selected_backend=args.wam_backend,
        use_provider_command=bool(args.use_provider_command),
        oscar_repo=args.oscar_repo,
        checkpoint=args.checkpoint,
        oscar_provider=args.oscar_provider,
        allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
    )
    write_json(out_dir / "closed_loop_wam_backend_readiness.json", wam_backend_readiness)
    oscar_ready = wam_backend_readiness["status"] == "ready"

    if args.dry_run or not oscar_ready:
        plan = {
            "schema_version": "oscar_isaac_closed_loop_plan.v1",
            "generated_at": utc_now_iso(),
            "status": "prepared" if oscar_ready else "blocked",
            "mode": "dry_run" if args.dry_run else "prepared",
            "start_frame": args.start_frame,
            "start_frame_present": Path(args.start_frame).expanduser().is_file(),
            "route_point_count": len(route),
            "projected_skeleton_trace_path": str(projected_skeleton_trace_path)
            if projected_skeleton_trace_path
            else None,
            "steps": int(args.steps),
            "task_prompt": args.task_prompt,
            "num_frames_per_step": int(args.num_frames),
            "oscar_runtime_settings": {
                "num_frames": int(args.num_frames),
                "num_steps": int(args.oscar_num_steps),
                "guidance": float(args.oscar_guidance),
                "seed": int(args.oscar_seed),
                "height": int(args.oscar_height),
                "width": int(args.oscar_width),
                "fps": float(args.oscar_fps),
            },
            "selected_wam_backend": args.wam_backend,
            "wam_backend_readiness_path": str(out_dir / "closed_loop_wam_backend_readiness.json"),
            "wam_backend_readiness": wam_backend_readiness,
            "use_provider_command": bool(args.use_provider_command),
            "oscar_provider": args.oscar_provider,
            "allow_paid_provider_launch": bool(args.allow_paid_provider_launch),
            "oscar_repo": args.oscar_repo,
            "checkpoint_configured": bool(args.checkpoint),
            "harness_backend_kind": args.harness_backend_kind,
            "harness_backend_command_argv0": harness_command[0] if harness_command else None,
            "perception_target_prompts": list(args.perception_target_prompt or []),
            "requirements": {
                "fresh_oscar_provider_required": bool(args.require_fresh_oscar_provider),
                "real_perception_backend_required": bool(args.require_real_perception_backend),
                "sam3_completed_required": bool(args.require_sam3_completed),
                "da3_completed_required": bool(args.require_da3_completed),
            },
            "blockers": list(wam_backend_readiness.get("blockers") or []),
        }
        write_json(out_dir / "oscar_isaac_closed_loop_plan.json", plan)
        print(json.dumps({"status": plan["status"], "mode": plan["mode"]}, sort_keys=True))
        return 0 if plan["status"] in {"prepared"} else 2

    if args.use_provider_command:
        backend = make_oscar_provider_command_wam_backend(
            work_dir=out_dir / "oscar_generation",
            task_prompt=args.task_prompt,
            num_frames=int(args.num_frames),
            num_steps=int(args.oscar_num_steps),
            guidance=float(args.oscar_guidance),
            seed=int(args.oscar_seed),
            height=int(args.oscar_height),
            width=int(args.oscar_width),
            fps=float(args.oscar_fps),
            provider=args.oscar_provider,
            allow_paid_provider_launch=bool(args.allow_paid_provider_launch),
            timeout_seconds=float(args.provider_timeout_seconds),
            projected_skeleton_trace_path=projected_skeleton_trace_path,
        )
    else:
        import subprocess

        oscar_generate = make_local_oscar_subprocess_generate(
            oscar_repo=args.oscar_repo,
            checkpoint=args.checkpoint,
            num_steps=int(args.oscar_num_steps),
            guidance=float(args.oscar_guidance),
            height=int(args.oscar_height),
            width=int(args.oscar_width),
            fps=float(args.oscar_fps),
            run=subprocess.run,
            extract_next_frame=extract_next_observation_frame_from_video,
        )
        backend = make_oscar_per_step_wam_backend(
            oscar_generate=oscar_generate,
            work_dir=out_dir / "oscar_generation",
            task_prompt=args.task_prompt,
            num_frames=int(args.num_frames),
            seed=int(args.oscar_seed),
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
        require_fresh_oscar_provider=bool(args.require_fresh_oscar_provider),
        require_real_perception_backend=bool(args.require_real_perception_backend),
        require_sam3_completed=bool(args.require_sam3_completed),
        require_da3_completed=bool(args.require_da3_completed),
        perception_target_prompts=list(args.perception_target_prompt or []),
        wam_backend_id=args.wam_backend,
        wam_backend_readiness=wam_backend_readiness,
    )
    print(json.dumps({"status": manifest["status"], "steps_executed": manifest.get("steps_executed")}, sort_keys=True))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
