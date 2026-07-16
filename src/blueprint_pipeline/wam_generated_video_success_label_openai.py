"""OpenAI-backed success labels for WAM-generated rollout videos.

The command consumes ``wam_success_label_request.json`` and writes
``wam_success_labels.command.json``. Labels are semantic judgments over sampled
generated-video frames only; they do not upgrade MuJoCo, controller, safety,
deployment, or physical-robot proof.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .wam_generated_video_success_label_gemini import (
    GENERATED_VIDEO_VLM_PROVENANCE,
    SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY,
    _bool_or_none,
    _confidence_or_none,
    _has_task_success_context,
    _rollout_task_prompt_record,
    _string,
    _task_success_criteria,
    attach_success_label_runtime_attestation,
)


GATE_ENV = "BLUEPRINT_ALLOW_OPENAI_WAM_SUCCESS_LABELING"
SHARED_GATE_ENV = "BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING"
MODEL_ENV = "BLUEPRINT_OPENAI_WAM_SUCCESS_LABEL_MODEL"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OUTPUT_FILENAME = "wam_success_labels.command.json"
DEFAULT_MAX_FRAMES = 5
DEFAULT_MAX_FRAME_DIMENSION = 768
DEFAULT_JPEG_QUALITY = 86
PROMPT_INSTRUCTION = (
    "You are judging sampled frames from a generated world-model rollout video for a "
    "robot manipulation task. Return compact JSON only. Judge whether the frames show "
    "realistic task success. Be strict: do not infer success from provider completion, "
    "scene motion, camera motion, or a valid video. If the robot does not visibly "
    "reach/contact the correct target, or if the target state change is not visibly "
    "caused by the robot, mark success false or null."
)
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_INSTRUCTION.encode("utf-8")).hexdigest()


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
    return f"openai_success_label_failed:{type(exc).__name__}"


def _video_sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return list(range(max(1, max_frames)))
    if max_frames <= 1:
        return [0]
    raw = [round(index * (frame_count - 1) / max(1, max_frames - 1)) for index in range(max_frames)]
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
        return [], ["missing_cv2_for_openai_wam_success_frames"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], ["generated_video_open_failed_for_success_frame_sampling"]
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
                    "evidence_ref": f"{video_path.resolve()}#frame={frame_index}",
                }
            )
            if len(frames) >= max_frames:
                break
        blockers = [] if frames else ["generated_video_success_frame_sampling_produced_no_frames"]
        return frames, blockers
    finally:
        capture.release()


def _openai_label_one(
    *,
    api_key: str,
    model: str,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    video_path: Path,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("missing_openai_package") from exc

    task_record = _rollout_task_prompt_record(request, rollout)
    task_prompt = _string(task_record.get("task_prompt"))
    success_criteria = _task_success_criteria(
        request=request,
        rollout=rollout,
        task_record=task_record,
    )
    prompt = {
        "instruction": PROMPT_INSTRUCTION,
        "required_json": {
            "scene_description": "one short sentence describing visible content",
            "success": "boolean or null",
            "confidence": "number from 0 to 1",
            "rationale": "one short sentence",
            "task_completion_evidence": ["short visual evidence"],
            "failure_modes": ["short failure evidence or empty list"],
            "end_effector_reaches_target": "boolean or null",
            "target_state_change_visible": "boolean or null",
            "robot_caused_target_motion": "boolean or null",
        },
        "task_prompt": task_prompt,
        "task_success_criteria": success_criteria,
        "rollout": {
            "rollout_id": rollout.get("rollout_id"),
            "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
            "policy_id": rollout.get("policy_id"),
            "model_rollout_confidence": rollout.get("model_rollout_confidence"),
            "generated_video_path": str(video_path),
            "sampled_frame_indices": [frame.get("frame_index") for frame in frames],
        },
        "claim_boundary": "This is only a semantic label on generated video frames, not physical robot proof.",
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
        max_output_tokens=900,
    )
    payload = _parse_json_text(_string(getattr(response, "output_text", "")) or "{}")
    if isinstance(payload.get("labels"), list) and payload["labels"]:
        first = payload["labels"][0]
        payload = dict(first) if isinstance(first, Mapping) else payload
    success = _bool_or_none(payload.get("success"))
    evidence_refs = [str(video_path.resolve())] + [
        str(frame.get("evidence_ref")) for frame in frames if frame.get("evidence_ref")
    ]
    criterion_results = [
        {
            "criterion_id": "end_effector_reaches_target",
            "passed": _bool_or_none(payload.get("end_effector_reaches_target")),
            "evidence_refs": evidence_refs,
        },
        {
            "criterion_id": "target_state_change_visible",
            "passed": _bool_or_none(payload.get("target_state_change_visible")),
            "evidence_refs": evidence_refs,
        },
        {
            "criterion_id": "robot_caused_target_motion",
            "passed": _bool_or_none(payload.get("robot_caused_target_motion")),
            "evidence_refs": evidence_refs,
        },
    ]
    return {
        "label_id": f"openai_{_string(rollout.get('rollout_id')) or 'rollout'}",
        "rollout_id": rollout.get("rollout_id"),
        "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
        "policy_id": rollout.get("policy_id"),
        "status": "success" if success is True else "failure" if success is False else "uncertain",
        "success": success,
        "confidence": _confidence_or_none(payload.get("confidence")),
        "rationale": _string(payload.get("rationale")) or None,
        "scene_description": _string(payload.get("scene_description")) or None,
        "task_completion_evidence": payload.get("task_completion_evidence")
        if isinstance(payload.get("task_completion_evidence"), list)
        else [],
        "failure_modes": payload.get("failure_modes")
        if isinstance(payload.get("failure_modes"), list)
        else [],
        "end_effector_reaches_target": _bool_or_none(payload.get("end_effector_reaches_target")),
        "target_state_change_visible": _bool_or_none(payload.get("target_state_change_visible")),
        "robot_caused_target_motion": _bool_or_none(payload.get("robot_caused_target_motion")),
        "task_success_criteria": success_criteria,
        "criterion_results": criterion_results,
        "evidence_refs": evidence_refs,
        "label_source": "openai_generated_video_frame_judge",
        "success_label_provenance": GENERATED_VIDEO_VLM_PROVENANCE,
        "success_label_claim_boundary": SUCCESS_RATE_GENERATED_VIDEO_VLM_CLAIM_BOUNDARY,
        "success_label_is_physics_or_captured_truth": False,
        "model": model,
        "visual_evidence_used": bool(frames),
        "sampled_frame_count": len(frames),
        "human_review_required": False,
        "human_review_recommended": True,
        "rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "safety_or_contact_validation_proven": False,
        "srcc_or_policy_ranking_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def build_openai_wam_success_labels(
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
        or os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT")
        or resolved_input.parent / DEFAULT_OUTPUT_FILENAME
    ).resolve()
    ensure_dir(resolved_output.parent)
    request = _load_json(resolved_input)
    model_name = _string(model or os.getenv(MODEL_ENV)) or DEFAULT_MODEL
    frame_limit = max_frames or int(
        _string(os.getenv("BLUEPRINT_OPENAI_WAM_SUCCESS_MAX_FRAMES")) or DEFAULT_MAX_FRAMES
    )
    frame_dimension = max_frame_dimension or int(
        _string(os.getenv("BLUEPRINT_OPENAI_WAM_SUCCESS_MAX_FRAME_DIMENSION"))
        or DEFAULT_MAX_FRAME_DIMENSION
    )
    blockers: list[str] = []
    labels: list[dict[str, Any]] = []
    sampled_rollouts: list[dict[str, Any]] = []
    if not (_truthy(os.getenv(GATE_ENV)) or _truthy(os.getenv(SHARED_GATE_ENV))):
        blockers.append(f"missing_env_{GATE_ENV}_or_{SHARED_GATE_ENV}")
    api_key, api_key_source = _api_key()
    if not api_key:
        blockers.append("missing_openai_api_key_or_key_file")
    rollouts = [
        dict(item) for item in request.get("rollouts", []) or [] if isinstance(item, Mapping)
    ][:max_rollouts]
    if not rollouts:
        blockers.append("missing_generated_rollouts")

    if not blockers:
        for rollout in rollouts:
            task_record = _rollout_task_prompt_record(request, rollout)
            if not _has_task_success_context(
                request=request,
                rollout=rollout,
                task_record=task_record,
            ):
                blockers.append(
                    "missing_task_prompt_or_task_success_metadata_for_generated_video_success_label"
                )
                continue
            ordered_rows = [
                dict(row)
                for row in rollout.get("ordered_step_videos", []) or []
                if isinstance(row, Mapping)
            ]
            video_texts = [_string(row.get("generated_video_path")) for row in ordered_rows] or [
                _string(rollout.get("generated_video_path"))
            ]
            video_paths: list[Path] = []
            for video_text in video_texts:
                video_path = Path(video_text).expanduser()
                if not video_path.is_absolute():
                    video_path = resolved_input.parent / video_path
                if not video_path.is_file():
                    blockers.append("generated_video_path_not_found")
                    video_paths = []
                    break
                video_paths.append(video_path)
            if not video_paths:
                continue
            frames: list[dict[str, Any]] = []
            frame_blockers: list[str] = []
            frames_per_clip = max(1, frame_limit // len(video_paths))
            for clip_index, video_path in enumerate(video_paths):
                clip_frames, clip_blockers = _sample_video_frames(
                    video_path=video_path,
                    max_frames=frames_per_clip,
                    max_dimension=max(1, frame_dimension),
                    jpeg_quality=DEFAULT_JPEG_QUALITY,
                )
                for frame in clip_frames:
                    frame["episode_clip_index"] = clip_index
                frames.extend(clip_frames)
                frame_blockers.extend(clip_blockers)
            video_path = video_paths[-1]
            sampled_rollouts.append(
                {
                    "rollout_id": rollout.get("rollout_id"),
                    "ordered_generated_video_paths": [str(path) for path in video_paths],
                    "full_ordered_episode_sampled": bool(
                        len(video_paths) == len(video_texts) and not frame_blockers
                    ),
                    "sampled_frame_count": len(frames),
                    "frame_blockers": frame_blockers,
                }
            )
            if frame_blockers:
                blockers.extend(frame_blockers)
                continue
            try:
                labels.append(
                    _openai_label_one(
                        api_key=api_key,
                        model=model_name,
                        request=request,
                        rollout=rollout,
                        video_path=video_path,
                        frames=frames,
                    )
                )
            except Exception as exc:  # pragma: no cover - live provider behavior
                blockers.append(_provider_error_blocker(exc))
                break

    manifest = {
        "schema_version": "wam_success_labels.command.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if labels and not blockers else "blocked",
        "provider": "openai",
        "model": model_name,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "api_key_configured": bool(api_key_source),
        "blockers": sorted(set(blockers)),
        "label_count": len(labels),
        "sampled_rollouts": sampled_rollouts,
        "labels": labels,
        "visual_evidence_used": bool(labels),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "success_label_provenance": GENERATED_VIDEO_VLM_PROVENANCE,
        "claim_boundary": {
            "success_label_is_from_generated_video_not_physical_robot": True,
            "success_label_does_not_prove_forward_inverse_consistency": True,
            "generated_world_policy_evaluation_scope_proven": False,
            "success_label_provenance": GENERATED_VIDEO_VLM_PROVENANCE,
            "success_rate_from_generated_video_vlm_is_not_physics_or_captured_truth": True,
        },
    }
    manifest = attach_success_label_runtime_attestation(
        manifest,
        inference_input_manifest_sha256=_string(request.get("inference_input_manifest_sha256")),
        output_dir=resolved_output.parent,
    )
    write_json(resolved_output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_INPUT"),
        required=not bool(os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_INPUT")),
    )
    parser.add_argument(
        "--output", type=Path, default=os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT")
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rollouts", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_openai_wam_success_labels(
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
                "label_count": result["label_count"],
                "blockers": result["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
