"""OpenAI-backed episode consistency labels for WAM-generated rollout videos.

The command consumes ``wam_episode_consistency_request.json`` and writes
``wam_episode_consistency.command.json``. Labels are external judgments over a
generated video plus trace context; they do not run WAM and do not upgrade task
success, safety, deployment, SRCC, or generated-world rank fidelity claims.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


GATE_ENV = "BLUEPRINT_ALLOW_OPENAI_WAM_EPISODE_CONSISTENCY"
MODEL_ENV = "BLUEPRINT_OPENAI_WAM_EPISODE_CONSISTENCY_MODEL"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_FILENAME = "wam_episode_consistency.command.json"
DEFAULT_MAX_FRAMES = 5
DEFAULT_MAX_FRAME_DIMENSION = 768
DEFAULT_JPEG_QUALITY = 85


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    return dict(value) if isinstance(value, Mapping) else {}


def _read_secret_file(env_name: str, default_path: str) -> tuple[str, str | None]:
    configured = _string(os.getenv(env_name))
    path = Path(configured or default_path).expanduser()
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value, env_name if configured else f"default_file:{path.name}"
    return "", None


def _api_key() -> tuple[str, str | None]:
    value = _string(os.getenv("OPENAI_API_KEY"))
    if value:
        return value, "env"
    for env_name, default_path in (
        ("OPENAI_API_KEY_FILE", "~/.blueprint-secrets/openai_api_key"),
        ("BLUEPRINT_OPENAI_API_KEY_FILE", "~/.blueprint-secrets/openai_api_key"),
    ):
        value, source = _read_secret_file(env_name, default_path)
        if value:
            return value, "file" if source else None
    return "", None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _string(value).lower()
    if text in {"true", "yes", "pass", "passed", "consistent"}:
        return True
    if text in {"false", "no", "fail", "failed", "inconsistent"}:
        return False
    return None


def _confidence_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(_string(value))))
    except ValueError:
        return None


def _provider_error_blocker(exc: Exception) -> str:
    text = str(exc).lower()
    if "insufficient_quota" in text or "quota" in text or "billing" in text:
        return "openai_quota_or_billing_blocked"
    if "rate limit" in text or "rate_limit" in text:
        return "openai_rate_limited"
    if "authentication" in text or "api key" in text or "401" in text:
        return "openai_authentication_failed"
    if "permission" in text or "403" in text:
        return "openai_permission_denied"
    return f"openai_wam_episode_consistency_failed:{type(exc).__name__}"


def _consistency_support_claim_boundary() -> dict[str, Any]:
    return {
        "consistency_label_is_external_to_wam_and_evaluator": True,
        "consistency_label_is_from_generated_video_and_trace_context": True,
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


def _rollout_task_prompt(request: Mapping[str, Any], rollout: Mapping[str, Any]) -> str:
    scenario_id = _string(rollout.get("scenario_eval_run_id"))
    for prompt in request.get("task_prompts", []) or []:
        if not isinstance(prompt, Mapping):
            continue
        if _string(prompt.get("scenario_eval_run_id")) == scenario_id:
            return _string(prompt.get("task_prompt"))
    return ""


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


def _sample_video_frames(
    *,
    video_path: Path,
    max_frames: int,
    max_dimension: int,
    jpeg_quality: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return [], ["missing_cv2_for_openai_wam_episode_consistency_frames"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], ["generated_video_open_failed_for_frame_sampling"]
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[dict[str, Any]] = []
        for frame_index in _video_sample_indices(frame_count, max_frames):
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            largest = max(height, width)
            if max_dimension > 0 and largest > max_dimension:
                scale = max_dimension / float(largest)
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                continue
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            frames.append(
                {
                    "frame_index": frame_index,
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                    "evidence_ref": f"{video_path}#frame={frame_index}",
                }
            )
            if len(frames) >= max_frames:
                break
        blockers = [] if frames else ["generated_video_frame_sampling_produced_no_frames"]
        return frames, blockers
    finally:
        capture.release()


def _openai_score_one(
    *,
    api_key: str,
    model: str,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("missing_openai_package") from exc

    prompt = {
        "instruction": (
            "You are an external evaluator, not the WAM. Judge whether sampled "
            "frames from a generated world-model rollout are forward/inverse "
            "consistent with the provided task prompt and trace summary. Return "
            "compact JSON only. Do not judge generated-world rank fidelity, robot "
            "deployment safety, or physical task success."
        ),
        "required_json": {
            "forward_consistent": "boolean or null",
            "inverse_consistent": "boolean or null",
            "confidence": "number from 0 to 1",
            "rationale": "one short sentence",
            "visible_action_alignment_evidence": ["short evidence"],
            "inconsistency_evidence": ["short evidence or empty list"],
        },
        "task_prompt": _rollout_task_prompt(request, rollout),
        "rollout": {
            "rollout_id": rollout.get("rollout_id"),
            "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
            "policy_id": rollout.get("policy_id"),
            "model_candidate": rollout.get("model_candidate"),
            "sampled_frame_indices": [frame.get("frame_index") for frame in frames],
        },
        "trace_summary": request.get("trace_summary"),
        "source_trace_paths": request.get("source_trace_paths"),
        "claim_boundary": (
            "This is an external consistency label on generated video frames and "
            "trace context only."
        ),
    }
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": json.dumps(prompt, sort_keys=True)}
    ]
    for frame in frames:
        image_url = _string(frame.get("image_url"))
        if image_url:
            content.append({"type": "input_image", "image_url": image_url})

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=800,
    )
    payload = _parse_json_text(_string(getattr(response, "output_text", "")) or "{}")
    if isinstance(payload.get("rollout_checks"), list) and payload["rollout_checks"]:
        first = payload["rollout_checks"][0]
        payload = dict(first) if isinstance(first, Mapping) else payload
    return {
        "rollout_id": rollout.get("rollout_id"),
        "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
        "policy_id": rollout.get("policy_id"),
        "model_candidate": rollout.get("model_candidate"),
        "forward_consistent": _bool_or_none(payload.get("forward_consistent")),
        "inverse_consistent": _bool_or_none(payload.get("inverse_consistent")),
        "confidence": _confidence_or_none(payload.get("confidence")),
        "rationale": _string(payload.get("rationale")) or None,
        "visible_action_alignment_evidence": payload.get("visible_action_alignment_evidence")
        if isinstance(payload.get("visible_action_alignment_evidence"), list)
        else [],
        "inconsistency_evidence": payload.get("inconsistency_evidence")
        if isinstance(payload.get("inconsistency_evidence"), list)
        else [],
        "evidence_refs": [
            ref for ref in (frame.get("evidence_ref") for frame in frames) if ref
        ],
        "label_source": "openai_wam_episode_consistency_judge",
        "model": model,
        "visual_evidence_used": bool(frames),
        "action_trace_evidence_used": True,
        "sampled_frame_count": len(frames),
        **_consistency_support_claim_boundary(),
        "task_success_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "safety_or_contact_validation_proven": False,
        "srcc_or_policy_ranking_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def build_openai_wam_episode_consistency_labels(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    model: str | None = None,
    max_rollouts: int = 5,
    max_frames: int | None = None,
    max_frame_dimension: int | None = None,
) -> dict[str, Any]:
    resolved_input = Path(input_path).resolve()
    resolved_output = Path(
        output_path
        or os.getenv("BLUEPRINT_WAM_CONSISTENCY_OUTPUT")
        or resolved_input.parent / DEFAULT_OUTPUT_FILENAME
    ).resolve()
    ensure_dir(resolved_output.parent)
    request = _load_json(resolved_input)
    model_name = _string(model or os.getenv(MODEL_ENV)) or DEFAULT_MODEL
    frame_limit = max_frames or int(
        _string(os.getenv("BLUEPRINT_OPENAI_WAM_EPISODE_CONSISTENCY_MAX_FRAMES"))
        or DEFAULT_MAX_FRAMES
    )
    frame_dimension = max_frame_dimension or int(
        _string(os.getenv("BLUEPRINT_OPENAI_WAM_EPISODE_CONSISTENCY_MAX_FRAME_DIMENSION"))
        or DEFAULT_MAX_FRAME_DIMENSION
    )
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    sampled_rollouts: list[dict[str, Any]] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    api_key, api_key_source = _api_key()
    if not api_key:
        blockers.append("missing_openai_api_key_or_key_file")
    rollouts = [
        dict(item)
        for item in request.get("rollouts", []) or []
        if isinstance(item, Mapping)
    ][:max_rollouts]
    if not rollouts:
        blockers.append("missing_generated_rollouts")

    if not blockers:
        for rollout in rollouts:
            video_text = _string(rollout.get("generated_video_path"))
            video_path = Path(video_text).expanduser()
            if not video_path.is_absolute():
                video_path = resolved_input.parent / video_path
            if not video_path.is_file():
                blockers.append("generated_video_path_not_found")
                continue
            frames, frame_blockers = _sample_video_frames(
                video_path=video_path,
                max_frames=max(1, frame_limit),
                max_dimension=max(1, frame_dimension),
                jpeg_quality=DEFAULT_JPEG_QUALITY,
            )
            sampled_rollouts.append(
                {
                    "rollout_id": rollout.get("rollout_id"),
                    "generated_video_path": str(video_path),
                    "sampled_frame_count": len(frames),
                    "frame_blockers": frame_blockers,
                }
            )
            if frame_blockers:
                blockers.extend(frame_blockers)
                continue
            try:
                checks.append(
                    _openai_score_one(
                        api_key=api_key,
                        model=model_name,
                        request=request,
                        rollout=rollout,
                        frames=frames,
                    )
                )
            except Exception as exc:  # pragma: no cover - live provider behavior
                blockers.append(_provider_error_blocker(exc))
                break

    manifest = {
        "schema_version": "wam_episode_consistency.command.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if checks and not blockers else "blocked",
        "provider": "openai_wam_episode_consistency_judge",
        "model": model_name,
        "api_key_configured": bool(api_key_source),
        "blockers": sorted(set(blockers)),
        "rollout_check_count": len(checks),
        "sampled_rollouts": sampled_rollouts,
        "rollout_checks": checks,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": _consistency_support_claim_boundary(),
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
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_openai_wam_episode_consistency_labels(
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
