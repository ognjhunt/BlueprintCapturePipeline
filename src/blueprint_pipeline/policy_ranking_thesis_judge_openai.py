"""Opt-in OpenAI judge for frozen policy-ranking WAM rollouts.

The adapter crops the released side-by-side OSCAR/RoboArena media before any
frame leaves the machine.  Only the generated left half is sampled.  The
benchmark labels and third-party physical right half are never included in a
provider request.  Provider use is disabled unless the explicit experiment gate
is set; constructing and validating the request inventory is offline.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_thesis import JUDGE_RESULT_SCHEMA, canonical_sha256, file_sha256


GATE_ENV = "BLUEPRINT_ALLOW_POLICY_RANKING_THESIS_OPENAI_JUDGE"
MODEL = "gpt-5-2025-08-07"
PROMPT = (
    "You are an independent, conservative evaluator of a generated robot world-model "
    "rollout. The images contain only the generated OSCAR half; benchmark labels and the "
    "real-robot half are withheld. Use the task instruction and temporal frames. Separate "
    "task progress from world-model validity. Do not reward attractive motion, provider "
    "completion, or an object change that is not visibly caused by the commanded robot "
    "skeleton/arm. Return JSON only with success_probability, progress_score_0_to_5, "
    "judge_confidence, action_following_confidence, temporal_coherence_confidence, "
    "critical_contradiction, abstain, and a short rationale. A score of 5 requires visible, "
    "stable task completion; 4 substantial near-completion; 3 target contact followed by a "
    "world-model failure; 2 task-directed approach; 1 model failure during approach; 0 no "
    "task-directed behavior. Lower confidence or abstain when the target is occluded, the "
    "skeleton and robot disagree, state jumps, progress is invented, or completion is ambiguous."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
MAX_DIMENSION = 768
JPEG_QUALITY = 86
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_MILLION_TOKENS = 1.25
OUTPUT_USD_PER_MILLION_TOKENS = 10.0
# Conservative admission allowance for one request. The real charge is recorded
# from response.usage and is normally much lower than this ceiling.
MAX_ESTIMATED_REQUEST_USD = 0.09
LOW_DETAIL_IMAGE_TOKENS = 70
SAMPLING_CONTRACT = {
    "model_snapshot": MODEL,
    "reasoning_effort": REASONING_EFFORT,
    "max_output_tokens_including_reasoning": MAX_OUTPUT_TOKENS,
    "temperature": "not_requested_model_default",
    "top_p": "not_requested_model_default",
    "seed": "not_supported_by_this_responses_configuration",
    "image_detail": "low",
    "max_image_dimension": MAX_DIMENSION,
    "jpeg_quality": JPEG_QUALITY,
    "response_format": "strict_json_schema",
    "store": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
        "progress_score_0_to_5": {"type": "integer", "minimum": 0, "maximum": 5},
        "judge_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "action_following_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "temporal_coherence_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "critical_contradiction": {"type": "boolean"},
        "abstain": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "success_probability",
        "progress_score_0_to_5",
        "judge_confidence",
        "action_following_confidence",
        "temporal_coherence_confidence",
        "critical_contradiction",
        "abstain",
        "rationale",
    ],
}


class JudgeResponseError(RuntimeError):
    """A provider response existed but did not contain a usable structured score."""

    def __init__(self, reason: str, response: Any):
        super().__init__(reason)
        incomplete = getattr(response, "incomplete_details", None)
        self.safe_details = {
            "reason": reason,
            "response_id": str(getattr(response, "id", "") or ""),
            "response_status": str(getattr(response, "status", "") or ""),
            "incomplete_reason": str(getattr(incomplete, "reason", "") or ""),
            "usage": _usage(response),
            "raw_response_persisted": False,
        }


def evaluator_digest(protocol: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "adapter": "policy_ranking_thesis_openai_judge.v1",
            "model": MODEL,
            "prompt_sha256": PROMPT_SHA256,
            "full_temporal_frames": 32,
            "cheap_baseline_frames": 2,
            "generated_crop": [0.0, 0.5],
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "sampling_contract": SAMPLING_CONTRACT,
            "output_schema": OUTPUT_SCHEMA,
            "thresholds": protocol.get("thresholds"),
        }
    )


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    value = json.loads(cleaned[start : end + 1] if start >= 0 and end >= start else cleaned)
    return dict(value) if isinstance(value, Mapping) else {}


def _frame_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if count == 2:
        return [0, frame_count - 1] if frame_count > 1 else [0]
    raw = [round(index * (frame_count - 1) / (count - 1)) for index in range(count)]
    return list(dict.fromkeys(raw))


def sample_generated_half(
    video_path: str | Path,
    *,
    frame_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Crop the generated left half before JPEG encoding or provider transport."""

    import cv2  # type: ignore[import-not-found]

    resolved = Path(video_path).resolve()
    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    frames: list[dict[str, Any]] = []
    source_width = source_height = 0
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for frame_index in _frame_indices(count, frame_limit):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"video_frame_decode_failed:{frame_index}")
            source_height, source_width = frame.shape[:2]
            if source_width < 2:
                raise ValueError("video_width_too_small_for_half_crop")
            cropped = frame[:, : source_width // 2]
            height, width = cropped.shape[:2]
            largest = max(height, width)
            if largest > MAX_DIMENSION:
                scale = MAX_DIMENSION / largest
                cropped = cv2.resize(
                    cropped,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if not ok:
                raise ValueError(f"jpeg_encode_failed:{frame_index}")
            frames.append(
                {
                    "frame_index": frame_index,
                    "image_url": "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            )
    finally:
        capture.release()
    return frames, {
        "source_video_sha256": file_sha256(resolved),
        "source_width": source_width,
        "source_height": source_height,
        "crop_x_pixels": [0, source_width // 2],
        "third_party_physical_pixels_encoded": False,
        "sampled_frame_indices": [item["frame_index"] for item in frames],
    }


def build_request_inventory(
    index: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    rollout_root: str | Path,
    partition: str,
) -> dict[str, Any]:
    sessions = set(protocol["partitions"][partition])
    root = Path(rollout_root).resolve()
    digest = evaluator_digest(protocol)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for source in index.get("rows", []):
        if not isinstance(source, Mapping) or source.get("session_id") not in sessions:
            continue
        video_path = root / str(source["relative_path"])
        if not video_path.is_file():
            blockers.append(f"video_missing:{source.get('session_id')}:{source.get('policy_id')}")
            continue
        if video_path.read_bytes()[:42].startswith(b"version https://git-lfs.github.com"):
            blockers.append(f"video_not_materialized:{source.get('session_id')}:{source.get('policy_id')}")
            continue
        actual_sha = file_sha256(video_path)
        if actual_sha != source.get("sha256"):
            blockers.append(f"video_digest_mismatch:{source.get('session_id')}:{source.get('policy_id')}")
            continue
        for method, frame_count in (
            (protocol["evaluator"]["full_temporal_method"], 32),
            (protocol["evaluator"]["cheap_baseline_method"], 2),
        ):
            rows.append(
                {
                    "session_id": source["session_id"],
                    "policy_id": source["policy_id"],
                    "task_instruction": source["language_instruction"],
                    "video_path": str(video_path),
                    "video_sha256": actual_sha,
                    "method": method,
                    "frame_count": frame_count,
                    "evaluator_digest": digest,
                    "benchmark_labels_included": False,
                    "third_party_physical_pixels_included": False,
                }
            )
            rows[-1]["request_id"] = canonical_sha256(rows[-1])
    expected = len(sessions) * len(protocol["policies"]) * 2
    if len(rows) != expected:
        blockers.append(f"request_count_expected_{expected}_got_{len(rows)}")
    image_tokens = sum(int(row["frame_count"]) * LOW_DETAIL_IMAGE_TOKENS for row in rows)
    # This is a deliberately conservative pre-call bound, not provider metering.
    # One token per three UTF-8 characters overstates typical English prompt tokenization.
    text_tokens_bound = sum(
        math.ceil((len(PROMPT) + len(str(row["task_instruction"])) + 512) / 3)
        for row in rows
    )
    output_tokens_bound = len(rows) * MAX_OUTPUT_TOKENS
    total_cost_bound = (
        (image_tokens + text_tokens_bound) * INPUT_USD_PER_MILLION_TOKENS
        + output_tokens_bound * OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000
    result: dict[str, Any] = {
        "schema_version": "policy_ranking_judge_request_inventory.v1",
        "status": "ready" if not blockers else "blocked",
        "partition": partition,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluator_digest": digest,
        "provider": "openai",
        "model": MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "sampling_contract": SAMPLING_CONTRACT,
        "request_count": len(rows),
        "precall_cost_bound": {
            "basis": "official_gpt_5_standard_rates_and_70_tokens_per_low_detail_image",
            "image_tokens": image_tokens,
            "text_tokens_conservative_bound": text_tokens_bound,
            "output_tokens_max": output_tokens_bound,
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
            "estimated_total_usd_upper_bound": total_cost_bound,
            "provider_metered_usage_available": False,
        },
        "requests": rows,
        "blockers": sorted(set(blockers)),
        "provider_called": False,
        "data_uploaded": False,
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = float(payload[key])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"out_of_range:{key}")
    return value


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cached = int(
        getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    )
    conservative_cost = (
        input_tokens * INPUT_USD_PER_MILLION_TOKENS
        + output_tokens * OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "estimated_cost_usd_conservative": conservative_cost,
    }


def _score_one(client: Any, request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    frames, crop = sample_generated_half(
        request["video_path"], frame_limit=int(request["frame_count"])
    )
    prompt = {
        "instruction": PROMPT,
        "task_instruction": request["task_instruction"],
        "method": request["method"],
        "sampled_frame_indices": crop["sampled_frame_indices"],
        "claim_boundary": "Generated-video judgment only; not physical or site-specific proof.",
    }
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": json.dumps(prompt, sort_keys=True)}
    ]
    content.extend(
        {"type": "input_image", "image_url": frame["image_url"], "detail": "low"}
        for frame in frames
    )
    started = time.monotonic()
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "policy_ranking_episode_judgment",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    elapsed = time.monotonic() - started
    try:
        payload = _parse_json(str(getattr(response, "output_text", "")))
    except (json.JSONDecodeError, ValueError) as exc:
        raise JudgeResponseError("unparseable_structured_output", response) from exc
    judgment = {
        "schema_version": JUDGE_RESULT_SCHEMA,
        "session_id": request["session_id"],
        "policy_id": request["policy_id"],
        "method": request["method"],
        "success_probability": _number(payload, "success_probability"),
        "progress_score_0_to_5": int(payload["progress_score_0_to_5"]),
        "judge_confidence": _number(payload, "judge_confidence"),
        "action_following_confidence": _number(payload, "action_following_confidence"),
        "temporal_coherence_confidence": _number(payload, "temporal_coherence_confidence"),
        "critical_contradiction": bool(payload.get("critical_contradiction")),
        "abstained": bool(payload.get("abstain")),
        "rationale": str(payload.get("rationale") or "")[:500],
        "evaluator_digest": request["evaluator_digest"],
        "benchmark_labels_seen": False,
        "third_party_physical_pixels_seen": False,
        "source_video_sha256": crop["source_video_sha256"],
        "crop_attestation": crop,
        "provider": "openai",
        "model": MODEL,
        "request_id": request.get("request_id") or canonical_sha256(dict(request)),
        "response_id": str(getattr(response, "id", "") or ""),
        "response_status": str(getattr(response, "status", "") or ""),
        "usage": _usage(response),
        "wall_time_seconds": elapsed,
    }
    return judgment, crop


def run_inventory(
    inventory: Mapping[str, Any],
    *,
    output_path: str | Path,
    max_requests: int | None = None,
    max_estimated_cost_usd: float = 2.0,
    max_workers: int = 4,
    max_attempts_per_request: int = 2,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        result = {
            "schema_version": "policy_ranking_judge_run.v1",
            "status": "blocked",
            "blockers": [f"missing_env_{GATE_ENV}"],
            "judgments": [],
            "provider_called": False,
            "data_uploaded": False,
        }
        write_json(Path(output_path), result)
        return result
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        result = {
            "schema_version": "policy_ranking_judge_run.v1",
            "status": "blocked",
            "blockers": ["missing_openai_api_key"],
            "judgments": [],
            "provider_called": False,
            "data_uploaded": False,
        }
        write_json(Path(output_path), result)
        return result
    from openai import OpenAI  # type: ignore[import-not-found]

    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers_must_be_between_1_and_8")
    if not 1 <= max_attempts_per_request <= 3:
        raise ValueError("max_attempts_per_request_must_be_between_1_and_3")

    client = OpenAI(api_key=key)
    requests = list(inventory.get("requests", []))
    if max_requests is not None:
        requests = requests[:max_requests]
    output_path = Path(output_path)
    previous: dict[str, Any] = {}
    if output_path.is_file():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if loaded.get("inventory_sha256") == inventory.get("inventory_sha256"):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}
    selected_ids = {str(row.get("request_id")) for row in requests}
    judgments = [
        dict(row)
        for row in previous.get("judgments", [])
        if isinstance(row, Mapping) and str(row.get("request_id")) in selected_ids
    ]
    completed_ids = {str(row.get("request_id")) for row in judgments}
    failed_requests = [
        dict(row)
        for row in previous.get("failed_requests", [])
        if isinstance(row, Mapping) and str(row.get("request_id")) in selected_ids
    ]
    blockers: list[str] = []
    started = time.monotonic()
    provider_called = bool(previous.get("provider_called"))
    uploaded = bool(previous.get("data_uploaded"))
    estimated_cost = sum(
        float((row.get("usage") or {}).get("estimated_cost_usd_conservative") or 0.0)
        for row in judgments
    )
    estimated_cost += sum(
        float((row.get("usage") or {}).get("estimated_cost_usd_conservative") or 0.0)
        for row in failed_requests
    )
    attempt_counts: dict[str, int] = defaultdict(int)
    for row in [*judgments, *failed_requests]:
        attempt_counts[str(row.get("request_id"))] += 1
    request_order = {
        str(request.get("request_id")): index for index, request in enumerate(requests)
    }
    pending = [
        request
        for request in requests
        if str(request.get("request_id")) not in completed_ids
        and attempt_counts[str(request.get("request_id"))] < max_attempts_per_request
    ]
    exhausted_ids = {
        str(request.get("request_id"))
        for request in requests
        if str(request.get("request_id")) not in completed_ids
        and attempt_counts[str(request.get("request_id"))] >= max_attempts_per_request
    }
    blockers.extend(f"retry_exhausted:{request_id}" for request_id in sorted(exhausted_ids))
    remaining_allowance = max(0.0, max_estimated_cost_usd - estimated_cost)
    admitted_count = min(
        len(pending), int(remaining_allowance // MAX_ESTIMATED_REQUEST_USD)
    )
    admitted = pending[:admitted_count]
    if admitted_count < len(pending):
        blockers.append("estimated_cost_cap_would_be_exceeded")

    def persist_checkpoint() -> None:
        judgments.sort(key=lambda row: request_order.get(str(row.get("request_id")), len(requests)))
        failed_requests.sort(
            key=lambda row: request_order.get(str(row.get("request_id")), len(requests))
        )
        checkpoint = {
            "schema_version": "policy_ranking_judge_run.v1",
            "status": "running",
            "inventory_sha256": inventory.get("inventory_sha256"),
            "provider": "openai",
            "model": MODEL,
            "sampling_contract": SAMPLING_CONTRACT,
            "request_count": len(requests),
            "judgment_count": len(judgments),
            "judgments": judgments,
            "failed_requests": failed_requests,
            "blockers": blockers,
            "provider_called": provider_called,
            "data_uploaded": uploaded,
            "estimated_cost_usd_conservative": estimated_cost,
            "max_estimated_cost_usd": max_estimated_cost_usd,
            "max_workers": max_workers,
            "max_attempts_per_request": max_attempts_per_request,
            "raw_credentials_written": False,
        }
        write_json(output_path, checkpoint)

    if admitted:
        provider_called = True
        uploaded = True
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score_one, client, request): request for request in admitted}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    judgment, _ = future.result()
                    judgments.append(judgment)
                    completed_ids.add(str(judgment["request_id"]))
                    estimated_cost += float(
                        judgment["usage"]["estimated_cost_usd_conservative"]
                    )
                except Exception as exc:  # pragma: no cover - provider/runtime behavior
                    if isinstance(exc, JudgeResponseError):
                        failed = {
                            "request_id": request.get("request_id"),
                            "session_id": request.get("session_id"),
                            "policy_id": request.get("policy_id"),
                            **exc.safe_details,
                        }
                        failed_requests.append(failed)
                        attempt_counts[str(request.get("request_id"))] += 1
                        estimated_cost += float(
                            (failed.get("usage") or {}).get(
                                "estimated_cost_usd_conservative"
                            )
                            or 0.0
                        )
                    blockers.append(
                        f"request_failed:{request.get('session_id')}:"
                        f"{request.get('policy_id')}:{type(exc).__name__}"
                    )
                persist_checkpoint()
    result = {
        "schema_version": "policy_ranking_judge_run.v1",
        "status": "completed" if len(judgments) == len(requests) and not blockers else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "provider": "openai",
        "model": MODEL,
        "prompt_sha256": PROMPT_SHA256,
        "sampling_contract": SAMPLING_CONTRACT,
        "request_count": len(requests),
        "judgment_count": len(judgments),
        "judgments": judgments,
        "failed_requests": failed_requests,
        "blockers": blockers,
        "provider_called": provider_called,
        "data_uploaded": uploaded,
        "estimated_cost_usd_conservative": estimated_cost,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "max_workers": max_workers,
        "max_attempts_per_request": max_attempts_per_request,
        "wall_time_seconds": time.monotonic() - started,
        "raw_credentials_written": False,
    }
    result["run_sha256"] = canonical_sha256(result)
    write_json(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--index", required=True)
    inventory.add_argument("--protocol", required=True)
    inventory.add_argument("--rollout-root", required=True)
    inventory.add_argument("--partition", required=True, choices=("pilot", "calibration", "heldout"))
    inventory.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--inventory", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-requests", type=int)
    run.add_argument("--max-estimated-cost-usd", type=float, default=2.0)
    run.add_argument("--max-workers", type=int, default=4)
    run.add_argument("--max-attempts-per-request", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        index_payload = json.loads(Path(args.index).read_text())
        protocol_payload = json.loads(Path(args.protocol).read_text())
        result = build_request_inventory(
            index_payload,
            protocol_payload,
            rollout_root=args.rollout_root,
            partition=args.partition,
        )
        write_json(Path(args.output), result)
    else:
        result = run_inventory(
            json.loads(Path(args.inventory).read_text()),
            output_path=args.output,
            max_requests=args.max_requests,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_workers=args.max_workers,
            max_attempts_per_request=args.max_attempts_per_request,
        )
    return 0 if result["status"] in {"ready", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
