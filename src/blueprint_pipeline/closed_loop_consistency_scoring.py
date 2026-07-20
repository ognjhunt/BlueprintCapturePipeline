"""Per-step external consistency-scoring orchestration for closed-loop WAM runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .oscar_cosmos_wam_evaluator import (
    WAM_CONSISTENCY_COMMAND_ENV,
    WAM_CONSISTENCY_COMMAND_OUTPUT,
    WAM_CONSISTENCY_GATE_ENV,
    _env_truthy as _wam_consistency_env_truthy,
    _normalize_wam_episode_consistency,
    _run_wam_consistency_command,
    _unscored_wam_episode_consistency,
    _wam_consistency_blockers,
)
from .wam_generated_video_review import visual_smoke_generated_rollouts_for_review


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [_string(item) for item in value if _string(item)]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_state_values(value: Any) -> tuple[list[float], bool]:
    if isinstance(value, bool) or value is None:
        return [], False
    if isinstance(value, (int, float)):
        number = _finite_float(value)
        return ([number] if number is not None else []), number is None
    if isinstance(value, Mapping):
        values: list[float] = []
        invalid = False
        for child in value.values():
            child_values, child_invalid = _numeric_state_values(child)
            values.extend(child_values)
            invalid = invalid or child_invalid
        return values, invalid
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        invalid = False
        for child in value:
            child_values, child_invalid = _numeric_state_values(child)
            values.extend(child_values)
            invalid = invalid or child_invalid
        return values, invalid
    return [], True


def _wam_consistency_command(explicit_command: str | None) -> str:
    return _string(explicit_command) or _string(os.getenv(WAM_CONSISTENCY_COMMAND_ENV))


def _score_closed_loop_step_episode_consistency(
    *,
    output_dir: Path,
    generated_at: str,
    step_index: int,
    policy_id: str,
    source_frame_path: str,
    generated_frame_path: str,
    wam_output: Mapping[str, Any],
    action: Mapping[str, Any],
    action_history: Sequence[Mapping[str, Any]],
    task_prompts: Sequence[str],
    wam_consistency_command: str | None,
    allow_wam_consistency_scoring: bool,
    require_strict_action_aware_consistency: bool,
    timeout_seconds: float,
    visual_smoke_fn=visual_smoke_generated_rollouts_for_review,
) -> dict[str, Any]:
    step_dir = output_dir / "wam_episode_consistency" / f"step_{step_index:04d}"
    ensure_dir(step_dir)
    generated_video = _string(wam_output.get("generated_video_path"))
    rollouts = (
        [
            {
                "rollout_id": f"oscar_isaac_closed_loop_step_{step_index:04d}",
                "scenario_eval_run_id": f"isaac_closed_loop_step_{step_index:04d}",
                "policy_id": policy_id,
                "model_candidate": _string(wam_output.get("wam_backend")) or "oscar_wam",
                "generated_video_path": generated_video,
                "generated_frame_path": generated_frame_path,
                "source_observation_frame_path": source_frame_path,
                "step_index": step_index,
            }
        ]
        if generated_video and Path(generated_video).expanduser().is_file()
        else []
    )
    visual_smoke = visual_smoke_fn(
        rollouts=rollouts,
        output_dir=step_dir / "visual_smoke",
        generated_at=generated_at,
        require_review_quality_profile=False,
    )
    visual_rollout_useful = bool(
        _mapping(visual_smoke.get("claim_boundary")).get(
            "visual_rollout_useful_for_task_success_review"
        )
    )
    visual_smoke_path = step_dir / "wam_generated_rollout_visual_smoke.json"
    write_json(visual_smoke_path, visual_smoke)
    request_path = step_dir / "wam_episode_consistency_request.json"
    output_path = step_dir / WAM_CONSISTENCY_COMMAND_OUTPUT
    checks_path = step_dir / "wam_consistency_checks.json"
    task_prompt = next((prompt for prompt in task_prompts if prompt), "")
    action_sha256 = _canonical_sha256(action)
    action_source = _mapping(action.get("learned_policy_endpoint_action")) or dict(action)
    action_vector: list[float] = []
    action_vector_field: str | None = None
    for field in (
        "action_chunk",
        "sonic_action_chunk",
        "controller_action",
        "joint_targets",
        "sonic_latent_action",
    ):
        if field not in action_source:
            continue
        values, invalid = _numeric_state_values(action_source.get(field))
        if values and not invalid:
            action_vector = values
            action_vector_field = field
            break
    if not action_vector:
        fallback_values, _invalid = _numeric_state_values(
            {
                "root_position": action.get("root_position"),
                "root_yaw_radians": action.get("root_yaw_radians"),
            }
        )
        action_vector = fallback_values
        action_vector_field = "root_position_and_yaw_fallback"
    source_timing = _mapping(action_source.get("action_timing"))
    control_hz = _finite_float(source_timing.get("control_hz"))
    sample_period = _finite_float(source_timing.get("sample_period_seconds"))
    if sample_period is None and control_hz is not None and control_hz > 0:
        sample_period = 1.0 / control_hz
    if control_hz is None and sample_period is not None and sample_period > 0:
        control_hz = 1.0 / sample_period
    if action_vector_field == "root_position_and_yaw_fallback" and sample_period is None:
        # The deterministic closed loop advances at the fixed 20 ms cadence used
        # by run_oscar_isaac_closed_loop when it creates each action record.
        sample_period = 0.02
        control_hz = 50.0
    sim_time = _finite_float(action.get("sim_time_s"))
    if sim_time is None and sample_period is not None:
        sim_time = float(step_index) * sample_period
    action_units = [str(item) for item in action_source.get("action_units") or []]
    if action_vector_field == "root_position_and_yaw_fallback" and not action_units:
        action_units = ["m", "m", "m", "rad"]
    strict_request_blockers: list[str] = []
    if require_strict_action_aware_consistency:
        if action_vector_field == "root_position_and_yaw_fallback" or not action_vector:
            strict_request_blockers.append(
                "strict_action_consistency_exact_commanded_action_chunk_missing"
            )
        if len(action_units) != len(action_vector) or any(not unit for unit in action_units):
            strict_request_blockers.append(
                "strict_action_consistency_per_dimension_action_units_missing_or_mismatch"
            )
        if (
            control_hz is None
            or control_hz <= 0
            or sample_period is None
            or sample_period <= 0
            or sim_time is None
        ):
            strict_request_blockers.append(
                "strict_action_consistency_action_timing_missing_or_invalid"
            )
    controller_fk_state = _mapping(wam_output.get("skeleton_conditioning"))
    generated_state = _mapping(wam_output.get("generated_robot_state"))
    generated_motion_sha256 = _file_sha256(generated_video) if generated_video else ""
    strict_action_contract = {
        "schema_version": "strict_action_aware_consistency_contract.v1",
        "required": bool(require_strict_action_aware_consistency),
        "commanded_action_sha256": action_sha256,
        "commanded_action": dict(action),
        "commanded_action_vector": action_vector,
        "action_vector_field": action_vector_field,
        "action_dimension": len(action_vector),
        "action_unit": "per_dimension",
        "action_units": action_units,
        "action_timing": {
            "step_index": int(step_index),
            "sim_time_s": sim_time,
            "control_hz": control_hz,
            "sample_period_seconds": sample_period,
            "unit": "s",
        },
        "controller_fk_conditioning": controller_fk_state,
        "controller_fk_state_sha256": _canonical_sha256(controller_fk_state),
        "generated_robot_state": generated_state,
        "generated_state_sha256": _canonical_sha256(generated_state),
        "generated_motion_sha256": generated_motion_sha256,
        "request_validation_status": (
            "ready" if not strict_request_blockers else "blocked"
        ),
        "request_validation_blockers": strict_request_blockers,
    }
    request = {
        "schema_version": "wam_episode_consistency_request.v2",
        "generated_at": generated_at,
        "status": "ready_for_external_episode_scorer"
        if rollouts and visual_rollout_useful and not strict_request_blockers
        else "blocked_strict_action_contract"
        if strict_request_blockers
        else "blocked_generated_rollout_visual_quality"
        if rollouts
        else "blocked_missing_generated_rollout",
        "source_isaac_closed_loop_output_dir": str(output_dir),
        "generated_rollout_results": str(step_dir / "wam_generated_rollout_results.json"),
        "generated_rollout_visual_smoke": str(visual_smoke_path),
        "generated_rollout_visual_smoke_status": _string(visual_smoke.get("status")),
        "generated_rollout_visually_useful_for_success_review": visual_rollout_useful,
        "rollouts": rollouts,
        "task_prompts": [
            {
                "scenario_eval_run_id": f"isaac_closed_loop_step_{step_index:04d}",
                "task_prompt": task_prompt,
                "task_id": "isaac_g1_oscar_per_step_closed_loop",
            }
        ],
        "source_trace_paths": {
            "source_observation_frame": source_frame_path,
            "generated_next_observation_frame": generated_frame_path,
            "generated_next_observation_video": generated_video or None,
        },
        "trace_summary": {
            "action_row_count": len(action_history) + 1,
            "current_step_index": step_index,
            "current_action_type": action.get("action_type") or action.get("policy_action"),
        },
        "strict_action_aware_consistency": strict_action_contract,
        "expected_output_path": str(output_path),
        "consistency_label_contract": {
            "required_top_level_keys": ["rollout_checks"],
            "rollout_check_required_keys": [
                "rollout_id",
                "forward_consistent",
                "inverse_consistent",
                "confidence",
                "rationale",
                "commanded_action_sha256",
                "recovered_action",
                "recovered_action_sha256",
                "per_dimension_error",
                "per_dimension_uncertainty",
                "calibration_identity",
                "threshold",
                "action_timing",
                "evidence_refs",
                "termination_chunk",
                "action_units",
                "controller_fk_state_sha256",
                "generated_state_sha256",
                "generated_motion_sha256",
                "scorer_runtime_id",
                "provider_output_replay_used",
                "forward_result",
                "inverse_result",
            ],
            "outcome_states": ["passed", "failed", "abstained"],
            "failure_domains": [
                "model_abstention",
                "infrastructure_failure",
                "protocol_failure",
            ],
        },
        "claim_boundary": {
            "scorer_is_separate_from_wam_execution_and_evaluator": True,
            "scorer_input_is_generated_video_and_trace_context_not_physical_robot": True,
            "consistency_label_does_not_prove_task_success": True,
            "consistency_label_does_not_prove_generated_world_rank_fidelity": True,
            "raw_credentials_written_to_artifacts": False,
        },
    }
    write_json(request_path, request)
    write_json(
        step_dir / "wam_generated_rollout_results.json",
        {
            "schema_version": "isaac_closed_loop_wam_generated_rollout_results.v1",
            "generated_at": generated_at,
            "status": "completed" if rollouts else "blocked_missing_generated_rollout",
            "rollouts": rollouts,
            "blockers": [] if rollouts else ["missing_generated_video_for_wam_episode_consistency"],
            "claim_boundary": {
                "generated_video_is_not_task_success_proof": True,
                "generated_video_is_not_forward_inverse_consistency": True,
            },
        },
    )

    command = _wam_consistency_command(wam_consistency_command)
    consistency_blockers: list[str] = []
    command_result: dict[str, Any] | None = None
    command_payload: dict[str, Any] = {}
    if strict_request_blockers:
        consistency_blockers = list(strict_request_blockers)
    elif not rollouts:
        consistency_blockers = ["missing_generated_video_for_wam_episode_consistency"]
    elif not visual_rollout_useful:
        consistency_blockers = _string_list(visual_smoke.get("blockers")) or [
            "generated_rollout_not_visually_useful_for_consistency_proof"
        ]
    elif allow_wam_consistency_scoring or command:
        if not _wam_consistency_env_truthy(WAM_CONSISTENCY_GATE_ENV):
            consistency_blockers.append(f"missing_env_{WAM_CONSISTENCY_GATE_ENV}")
        if not allow_wam_consistency_scoring:
            consistency_blockers.append("missing_cli_allow_wam_consistency_scoring")
        if not command:
            consistency_blockers.append("missing_wam_episode_consistency_command")
        if not consistency_blockers:
            command_payload, command_result = _run_wam_consistency_command(
                command=command,
                input_path=request_path,
                output_path=output_path,
                timeout_seconds=timeout_seconds,
            )
            if command_result.get("status") != "completed":
                consistency_blockers.extend(
                    _string_list(command_result.get("blockers"))
                    or ["wam_episode_consistency_command_blocked"]
                )
    else:
        consistency_blockers = ["requires_external_wam_episode_consistency_scorer"]

    if command_payload and not consistency_blockers:
        consistency = _normalize_wam_episode_consistency(
            command_payload=command_payload,
            rollouts=rollouts,
            generated_at=generated_at,
            action_conditioned_video_rollout_generated=bool(rollouts),
            action_conditioned_video_rollout_available=bool(rollouts),
            provider_output_replay_used=False,
            success_label_generated=False,
            visual_smoke_status=_string(visual_smoke.get("status")),
            visual_rollout_useful=visual_rollout_useful,
            command_result=command_result,
            strict_action_contract=(
                strict_action_contract if require_strict_action_aware_consistency else None
            ),
        )
    else:
        consistency = _unscored_wam_episode_consistency(
            generated_at=generated_at,
            rollouts=rollouts,
            action_conditioned_video_rollout_generated=bool(rollouts),
            action_conditioned_video_rollout_available=bool(rollouts),
            provider_output_replay_used=False,
            success_label_generated=False,
            visual_smoke_status=_string(visual_smoke.get("status")),
            visual_rollout_useful=visual_rollout_useful,
            blockers=consistency_blockers,
            blocked_reason="blocked_missing_generated_rollout"
            if not rollouts
            else "blocked_generated_rollout_visual_quality"
            if not visual_rollout_useful
            else None,
        )
        if command_result is not None:
            consistency["command_result"] = command_result
    scoring_requested = bool(
        require_strict_action_aware_consistency or allow_wam_consistency_scoring or command
    )
    payload_status = _string(command_payload.get("status")).strip().lower()
    model_abstained = bool(
        payload_status == "abstained"
        or any(
            _string(row.get("outcome")).strip().lower() == "abstained"
            for row in command_payload.get("rollout_checks", []) or []
            if isinstance(row, Mapping)
        )
    )
    infrastructure_failure = bool(
        scoring_requested
        and command_result is not None
        and command_result.get("status") != "completed"
    )
    protocol_failure = bool(
        scoring_requested
        and not model_abstained
        and not infrastructure_failure
        and not consistency.get("forward_inverse_consistency_proven")
    )
    consistency["model_abstained"] = model_abstained
    consistency["infrastructure_failure"] = infrastructure_failure
    consistency["protocol_failure"] = protocol_failure
    consistency["scorer_outcome"] = (
        "model_abstention"
        if model_abstained
        else "infrastructure_failure"
        if infrastructure_failure
        else "protocol_failure"
        if protocol_failure
        else "passed"
        if consistency.get("forward_inverse_consistency_proven")
        else "not_run"
    )
    consistency["early_termination_recommended"] = bool(
        scoring_requested and not consistency.get("forward_inverse_consistency_proven")
    )
    consistency["scoring_requested"] = scoring_requested
    consistency["visual_smoke_path"] = str(visual_smoke_path)
    consistency["request_path"] = str(request_path)
    consistency["commanded_action_sha256"] = strict_action_contract[
        "commanded_action_sha256"
    ]
    consistency["generated_motion_sha256"] = strict_action_contract[
        "generated_motion_sha256"
    ]
    rollout_checks = [
        dict(row)
        for row in consistency.get("rollout_checks") or []
        if isinstance(row, Mapping)
    ]
    consistency["recovered_action_sha256"] = (
        rollout_checks[0].get("recovered_action_sha256") if rollout_checks else None
    )
    write_json(checks_path, consistency)
    return {
        "request": request,
        "consistency": consistency,
        "request_path": str(request_path),
        "checks_path": str(checks_path),
        "visual_smoke_path": str(visual_smoke_path),
        "forward_inverse_consistency_proven": bool(
            consistency.get("forward_inverse_consistency_proven")
        ),
        "forward_dynamics_consistency_proven": bool(
            consistency.get("forward_dynamics_consistency_proven")
        ),
        "inverse_dynamics_consistency_proven": bool(
            consistency.get("inverse_dynamics_consistency_proven")
        ),
        "external_episode_consistency_scorer_ran": bool(
            consistency.get("external_episode_consistency_scorer_ran")
        ),
        "scorer_outcome": consistency.get("scorer_outcome"),
        "model_abstained": bool(consistency.get("model_abstained")),
        "infrastructure_failure": bool(consistency.get("infrastructure_failure")),
        "protocol_failure": bool(consistency.get("protocol_failure")),
        "early_termination_recommended": bool(consistency.get("early_termination_recommended")),
        "commanded_action_sha256": consistency.get("commanded_action_sha256"),
        "generated_motion_sha256": consistency.get("generated_motion_sha256"),
        "recovered_action_sha256": consistency.get("recovered_action_sha256"),
        "blockers": _wam_consistency_blockers(consistency),
    }
