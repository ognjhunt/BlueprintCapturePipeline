"""Local CV episode consistency labels for WAM-generated rollout videos.

This command is a quota-free fallback scorer. It is deliberately weaker than a
hosted VLM judge: it checks video decodability, nonblank structure, and visible
temporal change, then emits the same external-scorer contract consumed by the
WAM consistency normalizer. It does not prove task success, safety, deployment,
or generated-world rank fidelity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


GATE_ENV = "BLUEPRINT_ALLOW_LOCAL_WAM_EPISODE_CONSISTENCY"
DEFAULT_MODEL = "local-cv-frame-delta-v1"
DEFAULT_OUTPUT_FILENAME = "wam_episode_consistency.command.json"
DEFAULT_MAX_FRAMES = 5
MIN_FRAME_COUNT = 2
MIN_EDGE_DENSITY = 0.001
MIN_MEAN_ABS_DIFF = 1.0


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _video_sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return list(range(max(1, max_frames)))
    if max_frames <= 1:
        return [0]
    raw = [
        round(index * (frame_count - 1) / max(1, max_frames - 1))
        for index in range(max_frames)
    ]
    indices: list[int] = []
    for value in raw:
        if value not in indices:
            indices.append(value)
    return indices


def _task_prompt(request: Mapping[str, Any], rollout: Mapping[str, Any]) -> str:
    scenario_id = _string(rollout.get("scenario_eval_run_id"))
    for item in request.get("task_prompts", []) or []:
        if isinstance(item, Mapping) and _string(item.get("scenario_eval_run_id")) == scenario_id:
            return _string(item.get("task_prompt"))
    return ""


def _trace_available(request: Mapping[str, Any]) -> bool:
    trace_summary = request.get("trace_summary")
    if isinstance(trace_summary, Mapping):
        if int(trace_summary.get("policy_call_count") or 0) > 0:
            return True
        if int(trace_summary.get("wam_transition_count") or 0) > 0:
            return True
    trace_paths = request.get("source_trace_paths")
    if isinstance(trace_paths, Mapping):
        return any(_string(value) for value in trace_paths.values())
    return False


def _score_video(video_path: Path, max_frames: int) -> tuple[dict[str, Any], list[str]]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {}, ["missing_cv2_for_local_wam_episode_consistency"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}, ["generated_video_open_failed_for_local_consistency"]
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[Any] = []
        frame_indices: list[int] = []
        for frame_index in _video_sample_indices(frame_count, max(2, max_frames)):
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
            frame_indices.append(frame_index)
            if len(frames) >= max(2, max_frames):
                break
    finally:
        capture.release()

    blockers: list[str] = []
    if frame_count and frame_count < MIN_FRAME_COUNT:
        blockers.append("generated_video_too_few_frames_for_local_consistency")
    if len(frames) < MIN_FRAME_COUNT:
        blockers.append("generated_video_sampling_too_few_frames_for_local_consistency")
    if blockers:
        return {
            "frame_count": frame_count,
            "fps": fps,
            "sampled_frame_count": len(frames),
            "sampled_frame_indices": frame_indices,
        }, blockers

    first = frames[0]
    last = frames[-1]
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    last_gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    mean_abs_diff = float(cv2.absdiff(first_gray, last_gray).mean())
    first_edges = cv2.Canny(first_gray, 50, 150)
    last_edges = cv2.Canny(last_gray, 50, 150)
    first_edge_density = float((first_edges > 0).mean())
    last_edge_density = float((last_edges > 0).mean())
    min_edge_density = min(first_edge_density, last_edge_density)
    if min_edge_density < MIN_EDGE_DENSITY:
        blockers.append("generated_video_local_edge_density_too_low")
    if mean_abs_diff < MIN_MEAN_ABS_DIFF:
        blockers.append("generated_video_local_temporal_change_too_low")
    return {
        "frame_count": frame_count,
        "fps": fps,
        "sampled_frame_count": len(frames),
        "sampled_frame_indices": frame_indices,
        "first_edge_density": round(first_edge_density, 6),
        "last_edge_density": round(last_edge_density, 6),
        "mean_abs_diff_first_to_last": round(mean_abs_diff, 6),
        "min_required_edge_density": MIN_EDGE_DENSITY,
        "min_required_mean_abs_diff": MIN_MEAN_ABS_DIFF,
    }, blockers


def _support_claim_boundary() -> dict[str, Any]:
    return {
        "consistency_label_is_external_to_wam_and_evaluator": True,
        "consistency_label_is_from_generated_video_and_trace_context": True,
        "local_cv_scorer_is_not_vlm_semantic_judge": True,
        "consistency_label_does_not_prove_task_success": True,
        "consistency_label_does_not_prove_generated_world_rank_fidelity": True,
        "forward_inverse_consistency_is_reliability_review_signal_only": True,
        "forward_inverse_consistency_does_not_upgrade_evaluator_bounded_policy_ranking": True,
        "forward_inverse_consistency_does_not_prove_policy_success": True,
        "forward_inverse_consistency_does_not_prove_task_success": True,
        "forward_inverse_consistency_does_not_prove_rank_fidelity": True,
        "forward_inverse_consistency_does_not_prove_deployment_readiness": True,
        "forward_inverse_consistency_does_not_prove_sensor_truth": True,
        "forward_inverse_consistency_is_not_external_validation": True,
        "consistency_metrics_are_support_signals_only": True,
        "evaluator_bounded_policy_ranking_upgraded_by_consistency": False,
        "policy_success_claimed_from_consistency": False,
        "task_success_claimed_from_consistency": False,
        "rank_fidelity_claimed_from_consistency": False,
        "deployment_readiness_claimed_from_consistency": False,
        "sensor_truth_claimed_from_consistency": False,
        "external_validation_claimed_from_consistency": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def build_local_wam_episode_consistency_labels(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    model: str | None = None,
    max_rollouts: int = 5,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> dict[str, Any]:
    resolved_input = Path(input_path).expanduser().resolve()
    resolved_output = Path(
        output_path
        or os.getenv("BLUEPRINT_WAM_CONSISTENCY_OUTPUT")
        or resolved_input.parent / DEFAULT_OUTPUT_FILENAME
    ).expanduser().resolve()
    ensure_dir(resolved_output.parent)
    request = _load_json(resolved_input)
    model_name = _string(model) or DEFAULT_MODEL
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    rollouts = [
        dict(item)
        for item in request.get("rollouts", []) or []
        if isinstance(item, Mapping)
    ][: max(1, max_rollouts)]
    if not rollouts:
        blockers.append("missing_generated_rollouts")
    trace_available = _trace_available(request)
    if not trace_available:
        blockers.append("missing_trace_context_for_local_consistency")

    for rollout in rollouts:
        rollout_blockers: list[str] = []
        video_text = _string(rollout.get("generated_video_path"))
        video_path = Path(video_text).expanduser()
        if not video_path.is_absolute():
            video_path = resolved_input.parent / video_path
        if not video_path.is_file():
            rollout_blockers.append("generated_video_path_not_found")
            metrics: dict[str, Any] = {}
        else:
            metrics, rollout_blockers = _score_video(video_path, max_frames=max(2, max_frames))
        blockers.extend(rollout_blockers)
        passed = not rollout_blockers and trace_available
        evidence = [
            f"Decoded {metrics.get('sampled_frame_count', 0)} sampled frames.",
            (
                "Mean absolute first/last frame difference "
                f"{metrics.get('mean_abs_diff_first_to_last')}."
            ),
            (
                "Edge densities first/last "
                f"{metrics.get('first_edge_density')}/{metrics.get('last_edge_density')}."
            ),
        ]
        checks.append(
            {
                "rollout_id": rollout.get("rollout_id"),
                "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
                "policy_id": rollout.get("policy_id"),
                "model_candidate": rollout.get("model_candidate"),
                "forward_consistent": passed,
                "inverse_consistent": passed,
                "confidence": 0.72 if passed else 0.0,
                "rationale": (
                    "Local CV scorer found decodable, nonblank, temporally changing video "
                    "with trace context present."
                    if passed
                    else "Local CV scorer could not prove decodable nonblank temporal change with trace context."
                ),
                "visible_action_alignment_evidence": evidence if passed else [],
                "inconsistency_evidence": rollout_blockers,
                "evidence_refs": [str(video_path)] if video_text else [],
                "visual_evidence_used": True,
                "action_trace_evidence_used": trace_available,
                "label_source": "local_cv_wam_episode_consistency_judge",
                "model": model_name,
                "task_prompt": _task_prompt(request, rollout),
                "local_cv_metrics": metrics,
                "local_cv_scorer_is_not_vlm_semantic_judge": True,
                "consistency_label_does_not_prove_task_success": True,
                "consistency_label_does_not_prove_generated_world_rank_fidelity": True,
                "task_success_proven": False,
                "policy_success_proven": False,
                "rank_fidelity_result_proven": False,
                "deployment_readiness_proven": False,
                "sensor_truth_proven": False,
                "external_validation_proven": False,
                "public_claim_upgrade_allowed": False,
            }
        )

    manifest = {
        "schema_version": "wam_episode_consistency.command.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if checks and not blockers else "blocked",
        "provider": "local_cv_wam_episode_consistency_judge",
        "model": model_name,
        "blockers": sorted(set(blockers)),
        "rollout_check_count": len(checks),
        "rollout_checks": checks,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _support_claim_boundary(),
    }
    write_json(resolved_output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=os.getenv("BLUEPRINT_WAM_CONSISTENCY_INPUT"),
        required=not bool(os.getenv("BLUEPRINT_WAM_CONSISTENCY_INPUT")),
    )
    parser.add_argument("--output", type=Path, default=os.getenv("BLUEPRINT_WAM_CONSISTENCY_OUTPUT"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rollouts", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args(argv)
    result = build_local_wam_episode_consistency_labels(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_rollouts=args.max_rollouts,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "provider": result["provider"],
                "model": result["model"],
                "rollout_check_count": result["rollout_check_count"],
                "blockers": result["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
