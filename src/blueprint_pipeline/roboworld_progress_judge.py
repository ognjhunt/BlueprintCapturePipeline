"""Producer for ``roboworld_progress_score.v1`` graded task-progress scores.

Blueprint already carries the whole consumer side of graded progress scoring --
rubric validation, five segment-aggregation strategies, the aggregation
ablation, and the blinded judge-calibration campaign, all in
:mod:`blueprint_pipeline.roboworld_evaluator`.  What it did not carry was
anything that *produces* a score for those consumers: every live judge in the
repository emits a binary success label.

That gap matters more than it looks.  The one ablation RoboWorld reports against
its correlation metric is the rubric itself -- binary success scores Spearman
0.922 where the graded 0--5 progress rubric scores 0.970 -- so the scoring
rubric, not the generator, is the cheapest demonstrated lever on rank fidelity.
A binary label also cannot separate "the policy failed" from "the world model
fell apart", which is the distinction the rubric exists to draw.

This module closes that gap in two halves:

* a **frame-sampling contract** that fails closed when an episode is sampled too
  sparsely to support graded scoring.  Six frames across a 25-second rollout is
  0.24 fps -- enough to guess a terminal state, structurally insufficient to
  localise where progress stopped, and far too coarse for the
  ``progress_then_regression_aware`` and ``stable_maintenance`` aggregation
  strategies the evaluator already implements; and
* a **conversion and validation path** that turns per-frame judge output into
  segment-level ``roboworld_progress_score.v1`` artifacts, each one checked
  against the frozen rubric and its criterion-scoped view authority before it is
  emitted.

Scores produced here are generated-media review evidence.  A score of 5 does not
prove physical task success, contact truth, safety, deployment readiness, or
rank fidelity against real outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json
from .roboworld_evaluator import (
    PROGRESS_SCORE_SCHEMA_VERSION,
    PROGRESS_STAGES,
    WORLD_MODEL_FAILURE_STAGES,
    build_default_progress_profile,
    canonical_sha256,
    validate_progress_score,
)


SAMPLING_CONTRACT_SCHEMA_VERSION = "roboworld_progress_frame_sampling_contract.v1"
JUDGE_REQUEST_SCHEMA_VERSION = "roboworld_progress_judge_request.v1"
JUDGE_RESULT_SCHEMA_VERSION = "roboworld_progress_judge_result.v1"

GATE_ENV = "BLUEPRINT_ALLOW_ROBOWORLD_PROGRESS_JUDGE"
JUDGE_COMMAND_ENV = "BLUEPRINT_ROBOWORLD_PROGRESS_JUDGE_COMMAND"
JUDGE_INPUT_ENV = "BLUEPRINT_ROBOWORLD_PROGRESS_JUDGE_INPUT"
JUDGE_OUTPUT_ENV = "BLUEPRINT_ROBOWORLD_PROGRESS_JUDGE_OUTPUT"

# Minimum temporal resolution for graded progress scoring.  Two samples per
# second resolves a grasp transition; a quarter of a sample per second does not.
MIN_PROGRESS_SAMPLE_FPS = 2.0
# Absolute floor regardless of episode length, so a very short clip still
# carries enough samples for a progress curve rather than a pair of endpoints.
MIN_PROGRESS_FRAME_COUNT = 24
# `stable_maintenance` credits a 5 only when the trailing sampled frames all
# remain at 5, so a segment must carry at least this many samples for that
# strategy to be computable at all.
MIN_SEGMENT_FRAME_COUNT = 4

PROMPT_INSTRUCTION = (
    "You are scoring a generated world-model rollout of a robot manipulation task "
    "against a frozen 0-5 task-progress rubric. "
    "Score each sampled frame for how far the task has progressed, and report "
    "separately whether the WORLD MODEL itself failed (objects vanishing, bodies "
    "interpenetrating, physically impossible motion) rather than the policy. "
    "These are different failures and must not be conflated: a policy that never "
    "approaches the target scores low with no model failure; a rollout where the "
    "box dissolves mid-grasp is a model failure regardless of policy quality. "
    "Use only the views you are told carry authority for each judgement. "
    "Abstain rather than guess when the evidence does not support a score. "
    "Return compact JSON only."
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_INSTRUCTION.encode("utf-8")).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any, *, minimum: int = 0, maximum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


def required_frame_count(duration_seconds: float) -> int | None:
    """Frames a rollout of this length needs for graded progress scoring."""

    duration = _number(duration_seconds)
    if duration is None or duration <= 0.0:
        return None
    return max(MIN_PROGRESS_FRAME_COUNT, int(math.ceil(duration * MIN_PROGRESS_SAMPLE_FPS)))


def build_frame_sampling_contract(
    *,
    duration_seconds: Any,
    sampled_frame_count: Any,
    segment_count: Any = 1,
    source_frame_count: Any = None,
) -> dict[str, Any]:
    """Judge whether a sampling plan can support graded progress scoring.

    Fails closed.  A sampling plan that cannot localise progress produces a
    number that looks like a progress score but behaves like a terminal-state
    guess, and the whole point of the graded rubric is the difference between
    those two things.
    """

    blockers: list[str] = []
    duration = _number(duration_seconds)
    sampled = _integer(sampled_frame_count, minimum=0)
    segments = _integer(segment_count, minimum=1) or 0
    source = _integer(source_frame_count, minimum=0) if source_frame_count is not None else None

    if duration is None or duration <= 0.0:
        blockers.append("progress_sampling_duration_seconds_missing_or_invalid")
    if sampled is None:
        blockers.append("progress_sampling_frame_count_missing_or_invalid")
    if segments < 1:
        blockers.append("progress_sampling_segment_count_missing_or_invalid")

    required = required_frame_count(duration) if duration else None
    achieved_fps = None
    if duration and duration > 0.0 and sampled is not None:
        achieved_fps = round(sampled / duration, 6)
    if required is not None and sampled is not None and sampled < required:
        blockers.append("progress_sampling_below_required_frame_count")
    if achieved_fps is not None and achieved_fps + 1e-9 < MIN_PROGRESS_SAMPLE_FPS:
        blockers.append("progress_sampling_below_minimum_sample_fps")
    if sampled is not None and segments >= 1:
        per_segment = sampled // segments if segments else 0
        if per_segment < MIN_SEGMENT_FRAME_COUNT:
            blockers.append("progress_sampling_segment_frame_count_below_minimum")
    if source is not None and sampled is not None and sampled > source:
        blockers.append("progress_sampling_exceeds_source_frame_count")
    if (
        source is not None
        and required is not None
        and source < required
    ):
        # The clip itself is too short to be sampled adequately; re-generating at
        # a higher frame rate is the fix, not sampling the same frames harder.
        blockers.append("progress_source_frame_count_cannot_support_rubric")

    blockers = sorted(set(blockers))
    return {
        "schema_version": SAMPLING_CONTRACT_SCHEMA_VERSION,
        "status": "adequate" if not blockers else "inadequate",
        "duration_seconds": duration,
        "sampled_frame_count": sampled,
        "source_frame_count": source,
        "segment_count": segments or None,
        "achieved_sample_fps": achieved_fps,
        "required_frame_count": required,
        "minimum_sample_fps": MIN_PROGRESS_SAMPLE_FPS,
        "minimum_frame_count": MIN_PROGRESS_FRAME_COUNT,
        "minimum_segment_frame_count": MIN_SEGMENT_FRAME_COUNT,
        "adequate_for_graded_progress": not blockers,
        "blockers": blockers,
        "claim_boundary": {
            "adequate_sampling_is_not_task_success": True,
            "adequate_sampling_is_not_rank_fidelity": True,
        },
    }


def frame_sample_indices(source_frame_count: int, sampled_frame_count: int) -> list[int]:
    """Evenly spaced sample indices across a clip, endpoints included."""

    total = _integer(source_frame_count, minimum=1)
    wanted = _integer(sampled_frame_count, minimum=1)
    if total is None or wanted is None:
        return []
    if wanted >= total:
        return list(range(total))
    if wanted == 1:
        return [0]
    return sorted(
        {round(index * (total - 1) / (wanted - 1)) for index in range(wanted)}
    )


def _rubric_by_score(profile: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row.get("score")): dict(row)
        for row in _rows(profile.get("rubric"))
        if isinstance(row.get("score"), int)
    }


def _segment_frame_scores(frame_scores: Sequence[Any], segment_count: int) -> list[list[int]]:
    values = [
        value
        for value in (_integer(item, minimum=0, maximum=5) for item in frame_scores)
        if value is not None
    ]
    if not values or segment_count < 1:
        return []
    size = len(values) / segment_count
    segments: list[list[int]] = []
    for index in range(segment_count):
        start = int(round(index * size))
        end = int(round((index + 1) * size)) if index + 1 < segment_count else len(values)
        segments.append(values[start:end])
    return [segment for segment in segments if segment]


def build_progress_scores(
    *,
    judge_result: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert per-frame judge output into validated progress-score artifacts.

    Every emitted score is run back through
    :func:`~blueprint_pipeline.roboworld_evaluator.validate_progress_score`, so a
    row that would not satisfy the frozen rubric or its criterion-scoped view
    authority is reported as blocked rather than written.
    """

    selected_profile = dict(profile or build_default_progress_profile())
    rubric = _rubric_by_score(selected_profile)
    blockers: list[str] = []

    if judge_result.get("schema_version") != JUDGE_RESULT_SCHEMA_VERSION:
        blockers.append("progress_judge_result_schema_missing_or_unsupported")
    rollout_id = _string(judge_result.get("rollout_id"))
    criterion_id = _string(judge_result.get("criterion_id"))
    if not rollout_id:
        blockers.append("progress_judge_result_rollout_id_missing")
    if not criterion_id:
        blockers.append("progress_judge_result_criterion_id_missing")

    sampling = _mapping(judge_result.get("frame_sampling_contract"))
    if sampling.get("schema_version") != SAMPLING_CONTRACT_SCHEMA_VERSION:
        blockers.append("progress_judge_result_sampling_contract_missing")
    elif sampling.get("adequate_for_graded_progress") is not True:
        # Refusing here is the point: a graded score computed from an inadequate
        # sample is indistinguishable in shape from a valid one downstream.
        blockers.append("progress_judge_result_sampling_inadequate")

    frame_scores = judge_result.get("frame_scores")
    if not isinstance(frame_scores, list) or not frame_scores:
        blockers.append("progress_judge_result_frame_scores_missing")
        frame_scores = []
    if any(_integer(value, minimum=0, maximum=5) is None for value in frame_scores):
        blockers.append("progress_judge_result_frame_scores_out_of_range")

    segment_count = _integer(judge_result.get("segment_count"), minimum=1) or 1
    view_evidence = _rows(judge_result.get("view_evidence"))
    if not view_evidence:
        blockers.append("progress_judge_result_view_evidence_missing")

    failure_stage = _string(judge_result.get("world_model_failure_stage")) or "none"
    failure_detected = judge_result.get("world_model_failure_detected")
    if not isinstance(failure_detected, bool):
        blockers.append("progress_judge_result_failure_detected_must_be_boolean")
    if failure_stage not in WORLD_MODEL_FAILURE_STAGES:
        blockers.append("progress_judge_result_failure_stage_invalid")

    confidence = _number(judge_result.get("judge_confidence"))
    abstained = judge_result.get("judge_abstained")
    if confidence is None or not 0.0 <= confidence <= 1.0:
        blockers.append("progress_judge_result_confidence_missing_or_out_of_range")
    if not isinstance(abstained, bool):
        blockers.append("progress_judge_result_abstained_must_be_boolean")

    digests = {
        field: _string(judge_result.get(field)).lower()
        for field in ("prompt_sha256", "judge_model_sha256", "calibration_set_sha256")
    }
    for field, value in digests.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            blockers.append(f"progress_judge_result_digest_missing_or_invalid:{field}")

    scores: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    if not blockers:
        for index, segment in enumerate(_segment_frame_scores(frame_scores, segment_count)):
            terminal = segment[-1]
            rubric_row = rubric.get(terminal, {})
            allowed = {
                _string(value)
                for value in rubric_row.get("allowed_world_model_failure_stages", [])
                if _string(value)
            }
            # The rubric constrains which model-failure stage may accompany a
            # given progress score; a judge stage outside that set is dropped to
            # the rubric's own required stage rather than silently accepted.
            if rubric_row.get("world_model_failure_required") is True:
                segment_failure_stage = _string(
                    rubric_row.get("world_model_failure_stage")
                ) or failure_stage
                segment_failure_detected = True
            elif failure_stage in allowed and failure_detected is True:
                segment_failure_stage = failure_stage
                segment_failure_detected = True
            else:
                segment_failure_stage = "none"
                segment_failure_detected = False
            candidate = {
                "schema_version": PROGRESS_SCORE_SCHEMA_VERSION,
                "profile_id": _string(selected_profile.get("profile_id")),
                "profile_sha256": _string(selected_profile.get("profile_sha256")),
                "rollout_id": rollout_id,
                "segment_index": index,
                "criterion_id": criterion_id,
                "task_progress_score": terminal,
                "policy_progress_stage": _string(rubric_row.get("policy_progress_stage")),
                "world_model_failure_stage": segment_failure_stage,
                "world_model_failure_detected": segment_failure_detected,
                "criterion_evidence_refs": [
                    _string(ref)
                    for ref in judge_result.get("criterion_evidence_refs", []) or []
                    if _string(ref)
                ],
                "judge_confidence": confidence,
                "judge_abstained": abstained,
                "abstention_reason": _string(judge_result.get("abstention_reason")) or None,
                "prompt_sha256": digests["prompt_sha256"],
                "judge_model_sha256": digests["judge_model_sha256"],
                "calibration_set_sha256": digests["calibration_set_sha256"],
                "view_evidence": view_evidence,
                "sampled_frame_scores": segment,
            }
            validated = validate_progress_score(candidate, profile=selected_profile)
            validations.append(validated)
            if validated.get("blockers"):
                blockers.append(f"progress_score_invalid:segment_{index}")
            else:
                scores.append(candidate)

    blockers = sorted(set(blockers))
    return {
        "schema_version": "roboworld_progress_score_batch.v1",
        "generated_at": utc_now_iso(),
        "status": "produced" if not blockers else "blocked",
        "profile_id": _string(selected_profile.get("profile_id")),
        "profile_sha256": _string(selected_profile.get("profile_sha256")),
        "rollout_id": rollout_id or None,
        "criterion_id": criterion_id or None,
        "segment_count": segment_count,
        "scores": scores,
        "score_validations": validations,
        "frame_sampling_contract": sampling or None,
        "blockers": blockers,
        "claim_boundary": {
            "progress_score_is_generated_media_review_evidence": True,
            "score_five_is_not_physical_task_success": True,
            "progress_score_is_not_rank_fidelity": True,
            "world_model_failure_stage_is_a_judge_opinion": True,
            "public_claim_upgrade_allowed": False,
        },
    }


def build_judge_request(
    *,
    rollout_id: str,
    criterion_id: str,
    task_instruction: str,
    frame_uris: Sequence[str],
    view_roles: Mapping[str, Sequence[str]],
    duration_seconds: float,
    segment_count: int = 1,
    source_frame_count: int | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a judge request with its sampling contract already evaluated."""

    selected_profile = dict(profile or build_default_progress_profile())
    frames = [_string(uri) for uri in frame_uris if _string(uri)]
    contract = build_frame_sampling_contract(
        duration_seconds=duration_seconds,
        sampled_frame_count=len(frames),
        segment_count=segment_count,
        source_frame_count=source_frame_count,
    )
    request = {
        "schema_version": JUDGE_REQUEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "rollout_id": _string(rollout_id),
        "criterion_id": _string(criterion_id),
        "task_instruction": _string(task_instruction),
        "profile_id": _string(selected_profile.get("profile_id")),
        "profile_sha256": _string(selected_profile.get("profile_sha256")),
        "rubric": selected_profile.get("rubric"),
        "progress_stages": list(PROGRESS_STAGES),
        "world_model_failure_stages": list(WORLD_MODEL_FAILURE_STAGES),
        "view_roles": {
            _string(view): sorted({_string(role) for role in roles if _string(role)})
            for view, roles in dict(view_roles).items()
            if _string(view)
        },
        "frame_uris": frames,
        "segment_count": int(segment_count),
        "frame_sampling_contract": contract,
        "prompt_instruction": PROMPT_INSTRUCTION,
        "prompt_sha256": PROMPT_TEMPLATE_SHA256,
        "ready": contract.get("adequate_for_graded_progress") is True,
    }
    request["request_sha256"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "generated_at"}
    )
    return request


def run_progress_judge_command(
    request: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Invoke the configured external judge command for a prepared request.

    The provider call is delegated to an operator-configured command rather than
    embedded here, matching how the other judge lanes reach a model.  The
    command is only run when the gate env is set, the request is marked ready by
    its sampling contract, and a command is configured; otherwise this returns a
    blocked result without spending anything.
    """

    blockers: list[str] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append("progress_judge_not_authorized")
    if request.get("ready") is not True:
        blockers.append("progress_judge_request_not_ready")
    command = _string(os.getenv(JUDGE_COMMAND_ENV))
    if not command:
        blockers.append("progress_judge_command_not_configured")
    if blockers:
        return {
            "schema_version": JUDGE_RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": sorted(set(blockers)),
        }

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "roboworld_progress_judge_input.json"
    output_path = root / "roboworld_progress_judge_output.json"
    write_json(input_path, dict(request))
    environment = dict(os.environ)
    environment[JUDGE_INPUT_ENV] = str(input_path)
    environment[JUDGE_OUTPUT_ENV] = str(output_path)
    completed = subprocess.run(
        command,
        shell=True,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        return {
            "schema_version": JUDGE_RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["progress_judge_command_failed"],
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        }
    result = _mapping(read_json_any(output_path))
    result.setdefault("schema_version", JUDGE_RESULT_SCHEMA_VERSION)
    result.setdefault("frame_sampling_contract", request.get("frame_sampling_contract"))
    result.setdefault("rollout_id", request.get("rollout_id"))
    result.setdefault("criterion_id", request.get("criterion_id"))
    result.setdefault("segment_count", request.get("segment_count"))
    result.setdefault("prompt_sha256", request.get("prompt_sha256"))
    return result


def _command_sampling(args: argparse.Namespace) -> int:
    contract = build_frame_sampling_contract(
        duration_seconds=args.duration_seconds,
        sampled_frame_count=args.sampled_frames,
        segment_count=args.segments,
        source_frame_count=args.source_frames,
    )
    if args.output:
        write_json(Path(args.output), contract)
    print(json.dumps(contract, sort_keys=True))
    return 0 if contract["adequate_for_graded_progress"] else 1


def _command_request(args: argparse.Namespace) -> int:
    payload = _mapping(read_json_any(Path(args.input)))
    request = build_judge_request(
        rollout_id=_string(payload.get("rollout_id")),
        criterion_id=_string(payload.get("criterion_id")),
        task_instruction=_string(payload.get("task_instruction")),
        frame_uris=payload.get("frame_uris", []) or [],
        view_roles=_mapping(payload.get("view_roles")),
        duration_seconds=payload.get("duration_seconds"),
        segment_count=_integer(payload.get("segment_count"), minimum=1) or 1,
        source_frame_count=payload.get("source_frame_count"),
    )
    write_json(Path(args.output), request)
    print(json.dumps({"path": args.output, "ready": request["ready"]}, sort_keys=True))
    return 0 if request["ready"] else 1


def _command_score(args: argparse.Namespace) -> int:
    result = _mapping(read_json_any(Path(args.input)))
    batch = build_progress_scores(judge_result=result)
    write_json(Path(args.output), batch)
    print(json.dumps({"path": args.output, "status": batch["status"]}, sort_keys=True))
    return 0 if batch["status"] == "produced" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Produce and validate RoboWorld-inspired task-progress scores"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sampling = sub.add_parser("check-sampling", help="evaluate a frame-sampling plan")
    sampling.add_argument("--duration-seconds", type=float, required=True)
    sampling.add_argument("--sampled-frames", type=int, required=True)
    sampling.add_argument("--segments", type=int, default=1)
    sampling.add_argument("--source-frames", type=int, default=None)
    sampling.add_argument("--output", default=None)
    sampling.set_defaults(func=_command_sampling)

    request = sub.add_parser("request", help="build a judge request")
    request.add_argument("--input", required=True)
    request.add_argument("--output", required=True)
    request.set_defaults(func=_command_request)

    score = sub.add_parser("score", help="convert judge output into progress scores")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(func=_command_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
