"""Distinct, fail-closed Cosmos 3 Edge experiment adapter.

This is intentionally separate from ``cosmos3_wam_command_adapter``: Edge does
not inherit Cosmos3-Nano/SC3 qualification. A pinned external worker executes
frozen, pipeline-derived cells in forward, inverse, and reasoning modes while
Blueprint owns the evidence and claim-boundary contracts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import ensure_dir, parse_bool, read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256, executable_identity, run_json_worker
from .local_capture import resolve_local_capture_context
from .nvidia_siggraph_policy import evaluate_stop_rules
from .nvidia_experiment_resource import (
    load_resource_closeout,
    load_resource_context,
    resource_stop_evidence,
)
from .wam_generated_video_review import validate_generated_mp4_for_review


ENABLE_ENV = "BLUEPRINT_ALLOW_COSMOS3_EDGE_EXPERIMENT"
EXPECTED_MODEL_ID = "nvidia/Cosmos3-Edge"
EXPECTED_PARAMETER_COUNT_BILLION = 4
MODES = ("forward_dynamics", "inverse_dynamics", "reasoning")
WORKER_SCHEMA = "cosmos3_edge_worker_result.v1"
ADAPTER_SCHEMA = "cosmos3_edge_wam_command_adapter.v1"
ADAPTER_ID = "blueprint_cosmos3_edge_experimental_command_adapter"
SUPPORTED_ACTION_DIMENSIONS = {
    "general_camera_motion": 9,
    "autonomous_vehicle": 9,
    "egocentric_motion": 57,
    "single_franka_robotiq": 10,
    "dual_franka_robotiq": 20,
    "agibot": 29,
    "ur": 10,
    "google_robot": 10,
    "widowx_250": 10,
    "umi": 10,
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _path_within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _load_mapping(path: Path) -> dict[str, Any]:
    loaded = read_json_any(path)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(loaded)


def validate_edge_configuration(configuration: Mapping[str, Any]) -> list[str]:
    """Enforce the July 20, 2026 NVIDIA model-card runtime envelope."""

    blockers: list[str] = []
    if _string(configuration.get("precision")).lower() != "bf16":
        blockers.append("edge_only_model_card_tested_precision_bf16")
    if _string(configuration.get("size")).lower() not in {"256p", "480p"}:
        blockers.append("edge_generation_size_must_be_256p_or_480p")
    try:
        fps = float(configuration.get("fps"))
    except (TypeError, ValueError):
        fps = 0.0
    if not 12.0 <= fps <= 30.0:
        blockers.append("edge_generation_fps_outside_12_to_30")
    try:
        frame_count = int(configuration.get("num_frames"))
    except (TypeError, ValueError):
        frame_count = 0
    if not 50 <= frame_count <= 150:
        blockers.append("edge_generation_frame_count_outside_50_to_150")
    try:
        action_length = int(configuration.get("action_sequence_length"))
    except (TypeError, ValueError):
        action_length = 0
    if not 16 <= action_length <= 400:
        blockers.append("edge_action_sequence_length_outside_16_to_400")
    embodiment = _string(configuration.get("action_embodiment")).lower()
    expected_dimension = SUPPORTED_ACTION_DIMENSIONS.get(embodiment)
    try:
        action_dimension = int(configuration.get("action_dimension"))
    except (TypeError, ValueError):
        action_dimension = 0
    if expected_dimension is None:
        blockers.append("edge_action_embodiment_not_in_model_card_supported_list")
    elif action_dimension != expected_dimension:
        blockers.append(
            f"edge_action_dimension_mismatch:{embodiment}:expected_{expected_dimension}:got_{action_dimension}"
        )
    try:
        reasoning_tokens = int(configuration.get("reasoning_max_tokens"))
    except (TypeError, ValueError):
        reasoning_tokens = 0
    if reasoning_tokens < 4096:
        blockers.append("edge_reasoning_max_tokens_below_model_card_recommendation_4096")
    try:
        reasoning_fps = float(configuration.get("reasoning_video_fps"))
    except (TypeError, ValueError):
        reasoning_fps = 0.0
    if reasoning_fps != 4.0:
        blockers.append("edge_reasoning_video_fps_must_match_recommended_4")
    return blockers


def _frozen_cells(
    manifest: Mapping[str, Any], *, manifest_path: Path, pipeline_root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    if manifest.get("frozen") is not True:
        blockers.append("benchmark_manifest_not_frozen")
    if manifest.get("privacy_safe_pipeline_derived") is not True:
        blockers.append("benchmark_manifest_not_privacy_safe_pipeline_derived")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        return [], [*blockers, "benchmark_manifest_cells_missing"]
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cells):
        cell = _mapping(raw)
        cell_id = _string(cell.get("cell_id"))
        raw_path = _string(cell.get("input_path"))
        path = Path(raw_path)
        path = path if path.is_absolute() else manifest_path.parent / path
        digest = _string(cell.get("input_sha256")).lower()
        if not cell_id or cell_id in seen:
            blockers.append(f"benchmark_cell_id_missing_or_duplicate:{index}")
        else:
            seen.add(cell_id)
        if not path.is_file() or not _path_within(path, pipeline_root):
            blockers.append(f"benchmark_cell_input_missing_or_outside_pipeline:{cell_id or index}")
        elif digest != sha256_file(path):
            blockers.append(f"benchmark_cell_input_digest_mismatch:{cell_id or index}")
        cells.append(
            {
                "cell_id": cell_id or f"cell_{index}",
                "input_path": str(path.resolve()),
                "input_sha256": sha256_file(path) if path.is_file() else None,
                "action_encoding": cell.get("action_encoding"),
                "accepted_anchor_id": cell.get("accepted_anchor_id"),
            }
        )
    return cells, blockers


def _normalize_attempt(
    *,
    raw_path: Path,
    execution: Mapping[str, Any],
    run_dir: Path,
    mode: str,
    cell: Mapping[str, Any],
    expected: Mapping[str, Any],
    repeat_index: int,
    video_validator: Callable[[str | Path], dict[str, Any]],
) -> dict[str, Any]:
    payload = _load_mapping(raw_path) if raw_path.is_file() else {}
    blockers: list[str] = []
    expected_identity = {
        "model_id": EXPECTED_MODEL_ID,
        "parameter_count_billion": EXPECTED_PARAMETER_COUNT_BILLION,
        "model_revision": expected["model_revision"],
        "code_revision": expected["code_revision"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "configuration_sha256": expected["configuration_sha256"],
    }
    if payload.get("schema_version") != WORKER_SCHEMA:
        blockers.append("edge_worker_schema_invalid")
    if payload.get("status") != "completed":
        blockers.append("edge_worker_status_not_completed")
    if payload.get("mode") != mode:
        blockers.append("edge_worker_mode_mismatch")
    for field, value in expected_identity.items():
        if payload.get(field) != value:
            blockers.append(f"edge_worker_identity_mismatch:{field}")
    if payload.get("input_sha256") != cell.get("input_sha256"):
        blockers.append("edge_worker_input_digest_mismatch")
    if (
        execution.get("exit_code") != 0
        or execution.get("timed_out")
        or execution.get("launch_error")
    ):
        blockers.append("edge_worker_execution_failed")
    outputs: list[dict[str, Any]] = []
    raw_outputs = payload.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        blockers.append("edge_worker_outputs_missing")
    for index, value in enumerate(raw_outputs if isinstance(raw_outputs, list) else []):
        output = _mapping(value)
        kind = _string(output.get("kind"))
        path = Path(_string(output.get("path")))
        path = path if path.is_absolute() else run_dir / path
        exists = path.is_file() and path.stat().st_size > 0
        if not exists or not _path_within(path, run_dir):
            blockers.append(f"edge_worker_output_invalid:{kind or index}")
        normalized = {
            "kind": kind or f"output_{index}",
            "path": str(path.resolve()),
            "sha256": sha256_file(path) if exists else None,
            "bytes": path.stat().st_size if exists else 0,
            "metadata": _mapping(output.get("metadata")),
        }
        if kind == "generated_video" and exists:
            normalized["generated_video_review_validation"] = video_validator(path)
            if normalized["generated_video_review_validation"].get("status") != "completed":
                blockers.append("edge_forward_generated_video_not_reviewable")
        outputs.append(normalized)
    required_kind = {
        "forward_dynamics": "generated_video",
        "inverse_dynamics": "action_inference",
        "reasoning": "reasoning_result",
    }[mode]
    if required_kind not in {item["kind"] for item in outputs}:
        blockers.append(f"edge_required_mode_output_missing:{required_kind}")
    return {
        "schema_version": "cosmos3_edge_mode_attempt.v1",
        "attempt_id": f"{cell['cell_id']}:{mode}:repeat_{repeat_index}",
        "cell_id": cell["cell_id"],
        "mode": mode,
        "repeat_index": repeat_index,
        "status": "completed" if not blockers else "blocked",
        "identity": expected_identity,
        "input_sha256": cell.get("input_sha256"),
        "outputs": outputs,
        "metrics": _mapping(payload.get("metrics")),
        "grounding": _mapping(payload.get("grounding")),
        "abstention": _mapping(payload.get("abstention")),
        "execution": dict(execution),
        "blockers": blockers,
    }


def run_cosmos3_edge_experiment(
    *,
    capture_root: str | Path,
    frozen_manifest_path: str | Path,
    worker_command: str | Sequence[str] | None,
    model_revision: str,
    code_revision: str,
    checkpoint_path: str | Path,
    configuration: Mapping[str, Any],
    modes: Sequence[str] = MODES,
    allow_experiment: bool = False,
    license_id: str = "OpenMDW-1.1",
    license_compatible: bool = False,
    timeout_seconds: int = 900,
    repeat_runs: int = 2,
    resource_context_path: str | Path | None = None,
    resource_closeout_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    video_validator: Callable[[str | Path], dict[str, Any]] = validate_generated_mp4_for_review,
) -> dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    output_dir = context.pipeline_root / "cosmos3_edge_experiment"
    ensure_dir(output_dir)
    manifest_path = Path(frozen_manifest_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = dict(configuration)
    config_sha = canonical_sha256(config)
    config_path = output_dir / "configuration.json"
    write_json(config_path, config)
    blockers: list[str] = []
    resource_context, resource_blockers = load_resource_context(resource_context_path)
    blockers.extend(resource_blockers)
    resource_closeout, closeout_blockers = load_resource_closeout(
        resource_context, resource_closeout_path
    )
    blockers.extend(closeout_blockers)
    if repeat_runs < 2:
        blockers.append("edge_repeat_runs_must_be_at_least_two")
    blockers.extend(validate_edge_configuration(config))
    selected_modes = list(dict.fromkeys(_string(value) for value in modes if _string(value)))
    if set(selected_modes) != set(MODES):
        blockers.append("edge_experiment_requires_forward_inverse_and_reasoning_modes")
    if not manifest_path.is_file() or not _path_within(manifest_path, context.pipeline_root):
        blockers.append("frozen_manifest_missing_or_outside_pipeline")
        manifest: dict[str, Any] = {}
    else:
        manifest = _load_mapping(manifest_path)
    cells, cell_blockers = _frozen_cells(
        manifest, manifest_path=manifest_path, pipeline_root=context.pipeline_root
    )
    blockers.extend(cell_blockers)
    checkpoint_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
    if not checkpoint_sha:
        blockers.append("edge_checkpoint_missing")
    if not model_revision or not code_revision:
        blockers.append("edge_model_or_code_revision_not_pinned")
    if not license_id or not license_compatible:
        blockers.append("license_not_verified_compatible")
    env_source = os.environ if env is None else env
    gate = bool(allow_experiment and parse_bool(env_source.get(ENABLE_ENV), default=False))
    if not gate:
        blockers.append(f"edge_experiment_requires_flag_and_{ENABLE_ENV}=true")
    if worker_command is None:
        blockers.append("edge_worker_command_not_configured")
    expected = {
        "model_revision": model_revision,
        "code_revision": code_revision,
        "checkpoint_sha256": checkpoint_sha,
        "configuration_sha256": config_sha,
    }
    request = {
        "schema_version": "cosmos3_edge_experiment_request.v1",
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "frozen_manifest_path": str(manifest_path),
        "frozen_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "cells": cells,
        "modes": selected_modes,
        "repeat_runs": repeat_runs,
        "model_identity": {
            "model_id": EXPECTED_MODEL_ID,
            "parameter_count_billion": EXPECTED_PARAMETER_COUNT_BILLION,
            "model_revision": model_revision,
            "code_revision": code_revision,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "license_id": license_id,
            "license_compatible": license_compatible,
            "inherits_cosmos3_nano_sc3_qualification": False,
            "model_card_release_date": "2026-07-20",
            "model_card_runtime_envelope_validated": not validate_edge_configuration(config),
            "g1_7d_action_encoding_listed_as_supported": False,
        },
        "configuration_path": str(config_path),
        "configuration_sha256": config_sha,
        "worker_identity": executable_identity(worker_command, env=env) if worker_command else {},
        "execution_policy": {
            "gate_satisfied": gate,
            "network_policy": "disabled",
            "paid_resource_allocation_performed_by_this_command": False,
            "timeout_seconds_per_attempt": timeout_seconds,
        },
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
        "blockers": blockers,
    }
    request["request_fingerprint"] = canonical_sha256(request)
    request_path = output_dir / "request.json"
    write_json(request_path, request)

    attempts: list[dict[str, Any]] = []
    if not blockers and worker_command is not None:
        for cell in cells:
            for mode in selected_modes:
                for repeat_index in range(1, repeat_runs + 1):
                    run_dir = (
                        output_dir / "attempts" / cell["cell_id"] / mode / f"repeat_{repeat_index}"
                    )
                    ensure_dir(run_dir)
                    raw_path = run_dir / "worker_result.json"
                    execution = run_json_worker(
                        command=worker_command,
                        replacements={
                            "input": cell["input_path"],
                            "input_sha256": str(cell["input_sha256"]),
                            "output": str(raw_path),
                            "output_dir": str(run_dir),
                            "config": str(config_path),
                            "configuration_sha256": config_sha,
                            "mode": mode,
                            "cell_id": cell["cell_id"],
                            "model_revision": model_revision,
                            "code_revision": code_revision,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": str(checkpoint_sha),
                        },
                        working_directory=run_dir,
                        output_directory=run_dir,
                        raw_report_path=raw_path,
                        timeout_seconds=timeout_seconds,
                        network_policy="disabled",
                        env=env,
                        log_prefix="edge_worker",
                    )
                    attempts.append(
                        _normalize_attempt(
                            raw_path=raw_path,
                            execution=execution,
                            run_dir=run_dir,
                            mode=mode,
                            cell=cell,
                            expected=expected,
                            repeat_index=repeat_index,
                            video_validator=video_validator,
                        )
                    )
    blockers.extend(
        f"attempt:{attempt['attempt_id']}:{blocker}"
        for attempt in attempts
        for blocker in attempt["blockers"]
    )
    all_attempts_expected = len(cells) * len(MODES) * repeat_runs
    if attempts and len(attempts) != all_attempts_expected:
        blockers.append("edge_attempt_matrix_incomplete")
    completed = bool(attempts) and not blockers
    stability_rows: list[dict[str, Any]] = []
    for cell in cells:
        for mode in selected_modes:
            group = [
                attempt
                for attempt in attempts
                if attempt["cell_id"] == cell["cell_id"] and attempt["mode"] == mode
            ]
            output_digests = [
                tuple(item.get("sha256") for item in attempt["outputs"]) for attempt in group
            ]
            stability_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "mode": mode,
                    "repeat_count": len(group),
                    "all_attempts_completed": bool(group)
                    and all(attempt["status"] == "completed" for attempt in group),
                    "exact_output_digest_stable": bool(output_digests)
                    and len(set(output_digests)) == 1,
                    "output_digest_sets": [list(value) for value in output_digests],
                }
            )
    attempt_manifest = {
        "schema_version": "cosmos3_edge_experiment_attempt_manifest.v1",
        "generated_at": utc_now_iso(),
        "status": "completed_advisory" if completed else "blocked",
        "request_path": request_path.name,
        "attempt_count": len(attempts),
        "expected_attempt_count": all_attempts_expected,
        "attempts": attempts,
        "output_stability": stability_rows,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "edge_model_execution_proven": completed,
            "cosmos3_nano_sc3_qualification_inherited": False,
            "structured_physics_truth_proven": False,
            "task_success_proven": False,
            "forward_inverse_consistency_proven": False,
            "rank_fidelity_result_proven": False,
            "real_world_correlation_proven": False,
            "default_model_change_allowed": False,
        },
    }
    attempt_manifest["manifest_fingerprint"] = canonical_sha256(attempt_manifest)
    attempt_path = output_dir / "attempt_manifest.json"
    write_json(attempt_path, attempt_manifest)

    forward_attempts = [attempt for attempt in attempts if attempt["mode"] == "forward_dynamics"]
    rollouts: list[dict[str, Any]] = []
    for attempt in forward_attempts:
        video = next((item for item in attempt["outputs"] if item["kind"] == "generated_video"), {})
        if video:
            rollouts.append(
                {
                    "rollout_id": attempt["attempt_id"],
                    "cell_id": attempt["cell_id"],
                    "generated_video_path": video.get("path"),
                    "generated_video_sha256": video.get("sha256"),
                    "generated_video_review_validation": video.get(
                        "generated_video_review_validation"
                    ),
                }
            )
    adapter_output = {
        "schema_version": ADAPTER_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "status": "completed" if completed and rollouts else "blocked",
        "blockers": []
        if completed and rollouts
        else list(dict.fromkeys(blockers or ["edge_forward_rollouts_missing"])),
        "learned_wam_model_ran": completed,
        "fresh_model_command_executed_this_invocation": completed,
        "fresh_model_run_claimed": completed,
        "fresh_model_run_steps": len(rollouts) if completed else 0,
        "configured_inference_steps_per_model_run": int(config.get("inference_steps", 1)),
        "edge_subprocess": {
            "status": "completed" if completed else "blocked",
            "returncode": 0 if completed else 2,
        },
        "model_provenance": {
            "model": EXPECTED_MODEL_ID,
            "model_revision": model_revision,
            "code_revision": code_revision,
            "checkpoint_sha256": checkpoint_sha,
            "parameter_count_billion": EXPECTED_PARAMETER_COUNT_BILLION,
        },
        "rollouts": rollouts,
        "mode_attempt_manifest_path": str(attempt_path),
        "truth_boundary": {
            "generated_video_is_model_output": completed and bool(rollouts),
            "forward_inverse_reasoning_modes_executed": completed,
            "physics_or_contact_correctness_proven": False,
            "task_success_proven": False,
            "rank_fidelity_result_proven": False,
        },
    }
    adapter_path = output_dir / "wam_command_adapter_result.json"
    write_json(adapter_path, adapter_output)
    stop_rules = evaluate_stop_rules(
        component="cosmos3_edge",
        require_measured_value=True,
        evidence={
            "component_version_pinned": bool(model_revision and code_revision and checkpoint_sha),
            "license_compatible": license_compatible,
            "stable_normalized_receipts": completed,
            "privacy_safe_inputs_only": bool(cells and not cell_blockers),
            "dependency_isolated": True,
            "input_output_digests_preserved": bool(completed and rollouts),
            "proof_boundaries_separated": True,
            "useful_failure_class_or_cost_gain": False,
            **resource_stop_evidence(resource_context, resource_closeout),
        },
    )
    result = {
        "schema_version": "cosmos3_edge_experiment_result.v1",
        "status": "completed_advisory" if completed else "blocked",
        "request_path": str(request_path),
        "attempt_manifest_path": str(attempt_path),
        "wam_command_adapter_result_path": str(adapter_path),
        "attempt_count": len(attempts),
        "output_stability": stability_rows,
        "resource_context": resource_context,
        "resource_closeout": resource_closeout or None,
        "stop_rule_evaluation": stop_rules,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": attempt_manifest["claim_boundary"],
    }
    result_path = output_dir / "result.json"
    write_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a gated Cosmos 3 Edge experiment")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--worker-command", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--license-id", default="OpenMDW-1.1")
    parser.add_argument("--license-compatible", action="store_true")
    parser.add_argument("--allow-experiment", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--repeat-runs", type=int, default=2)
    parser.add_argument("--resource-context")
    parser.add_argument("--resource-closeout")
    args = parser.parse_args(argv)
    result = run_cosmos3_edge_experiment(
        capture_root=args.capture_root,
        frozen_manifest_path=args.frozen_manifest,
        worker_command=args.worker_command,
        model_revision=args.model_revision,
        code_revision=args.code_revision,
        checkpoint_path=args.checkpoint,
        configuration=_load_mapping(Path(args.configuration)),
        license_id=args.license_id,
        license_compatible=args.license_compatible,
        allow_experiment=args.allow_experiment,
        timeout_seconds=args.timeout_seconds,
        repeat_runs=args.repeat_runs,
        resource_context_path=args.resource_context,
        resource_closeout_path=args.resource_closeout,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed_advisory" else 2


if __name__ == "__main__":
    raise SystemExit(main())
