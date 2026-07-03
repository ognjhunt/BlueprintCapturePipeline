"""Gemini-backed success labels for WAM-generated rollout videos.

The command consumes ``wam_success_label_request.json`` and writes
``wam_success_labels.command.json``. Labels are semantic judgments over
generated videos only; they do not upgrade MuJoCo, controller, safety,
deployment, or physical-robot proof.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


GATE_ENV = "BLUEPRINT_ALLOW_GEMINI_WAM_SUCCESS_LABELING"
MODEL_ENV = "BLUEPRINT_GEMINI_WAM_SUCCESS_LABEL_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OUTPUT_FILENAME = "wam_success_labels.command.json"
DEFAULT_MAX_INLINE_BYTES = 95 * 1024 * 1024


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    for env_name in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_AI_API_KEY"):
        value = _string(os.getenv(env_name))
        if value:
            return value, "env"
    for env_name, default_path in (
        ("GEMINI_API_KEY_FILE", "~/.blueprint-secrets/gemini_api_key"),
        ("GOOGLE_GENAI_API_KEY_FILE", "~/.blueprint-secrets/google_genai_api_key"),
        ("GOOGLE_AI_API_KEY_FILE", "~/.blueprint-secrets/google_ai_api_key"),
    ):
        value, source = _read_secret_file(env_name, default_path)
        if value:
            return value, "file" if source else None
    return "", None


def _rollout_task_prompt(request: Mapping[str, Any], rollout: Mapping[str, Any]) -> str:
    return _string(_rollout_task_prompt_record(request, rollout).get("task_prompt"))


def _rollout_task_prompt_record(
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_id = _string(rollout.get("scenario_eval_run_id"))
    for prompt in request.get("task_prompts", []) or []:
        if not isinstance(prompt, Mapping):
            continue
        if _string(prompt.get("scenario_eval_run_id")) == scenario_id:
            return dict(prompt)
    return {}


def _append_text_values(target: list[str], value: Any) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            target.append(text)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _append_text_values(target, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            _append_text_values(target, item)


def _metadata_list(payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        _append_text_values(values, payload.get(key))
    return values


def _metadata_success_requirements(payload: Mapping[str, Any]) -> list[str]:
    values = _metadata_list(
        payload,
        (
            "success_requires",
            "success_conditions",
            "strict_task_success_requirements",
            "target_state_change",
            "expected_state_change",
        ),
    )
    for key in ("task_success_criteria", "success_criteria", "success_check_plan"):
        container = _mapping(payload.get(key))
        if not container:
            continue
        values.extend(
            _metadata_list(
                container,
                (
                    "success_requires",
                    "criteria",
                    "checks",
                    "success_conditions",
                    "strict_task_success_requirements",
                    "target_state_change",
                    "expected_state_change",
                ),
            )
        )
    return values


def _metadata_failure_modes(payload: Mapping[str, Any]) -> list[str]:
    values = _metadata_list(
        payload,
        (
            "failure_modes",
            "common_failure_modes",
            "negative_criteria",
            "failure_conditions",
        ),
    )
    for key in ("task_success_criteria", "success_criteria", "success_check_plan"):
        container = _mapping(payload.get(key))
        if not container:
            continue
        values.extend(
            _metadata_list(
                container,
                (
                    "failure_modes",
                    "common_failure_modes",
                    "negative_criteria",
                    "failure_conditions",
                ),
            )
        )
    return values


def _task_success_criteria(
    *,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    task_record: Mapping[str, Any],
) -> dict[str, Any]:
    generic_criteria: list[str] = [
        "The visible robot end effector must reach the task-relevant target, not merely move near the scene.",
        "The target object or control must visibly change into the requested state.",
        "The state change must be causally plausible from the robot motion; object motion without visible robot contact/reach is failure or uncertain.",
        "If the relevant target is occluded, too small, or outside the frame, mark success null or false with a failure mode.",
    ]
    generic_failure_modes: list[str] = [
        "end_effector_does_not_reach_target",
        "target_state_change_not_visible",
        "target_motion_not_robot_caused",
        "target_occluded_or_out_of_frame",
    ]
    metadata_sources: dict[str, list[str]] = {}
    metadata_criteria: list[str] = []
    metadata_failure_modes: list[str] = []
    for source_name, payload in (
        ("request", request),
        ("request.success_label_contract", _mapping(request.get("success_label_contract"))),
        ("request.eval_ready_task_grounding", _mapping(request.get("eval_ready_task_grounding"))),
        ("task_prompt", task_record),
        (
            "task_prompt.eval_ready_task_grounding",
            _mapping(task_record.get("eval_ready_task_grounding")),
        ),
        ("rollout", rollout),
        ("rollout.eval_ready_task_grounding", _mapping(rollout.get("eval_ready_task_grounding"))),
    ):
        source_criteria = _metadata_success_requirements(payload)
        source_failure_modes = _metadata_failure_modes(payload)
        if source_criteria:
            metadata_sources.setdefault(source_name, []).extend(source_criteria)
            metadata_criteria.extend(source_criteria)
        if source_failure_modes:
            metadata_sources.setdefault(source_name, []).extend(source_failure_modes)
            metadata_failure_modes.extend(source_failure_modes)
    return {
        "success_requires": list(dict.fromkeys([*generic_criteria, *metadata_criteria])),
        "common_failure_modes": sorted(set([*generic_failure_modes, *metadata_failure_modes])),
        "metadata_sources": metadata_sources,
        "task_specific_rules_source": "request_or_rollout_metadata"
        if metadata_criteria or metadata_failure_modes
        else "generic_manipulation_evidence_only",
        "hardcoded_task_family_rules_used": False,
        "fail_closed_when_evidence_is_ambiguous": True,
    }


def _has_task_success_context(
    *,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    task_record: Mapping[str, Any],
) -> bool:
    criteria = _task_success_criteria(
        request=request,
        rollout=rollout,
        task_record=task_record,
    )
    metadata_sources = set(_mapping(criteria.get("metadata_sources")).keys())
    task_specific_metadata_sources = metadata_sources - {"request.success_label_contract"}
    return bool(
        _string(task_record.get("task_prompt"))
        or task_specific_metadata_sources
    )


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _string(value).lower()
    if text in {"true", "yes", "success", "succeeded", "pass", "passed"}:
        return True
    if text in {"false", "no", "failure", "failed", "fail"}:
        return False
    return None


def _confidence_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    try:
        return max(0.0, min(1.0, float(_string(value))))
    except ValueError:
        return None


def _provider_error_blocker(exc: Exception) -> str:
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "prepayment credits are depleted" in text:
        return "gemini_resource_exhausted_or_billing_credits_depleted"
    if "PERMISSION_DENIED" in text:
        return "gemini_permission_denied"
    if "INVALID_ARGUMENT" in text:
        return "gemini_invalid_argument"
    return f"gemini_success_label_failed:{type(exc).__name__}"


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.")
    return cleaned or "rollout"


def _extract_first_keyframe(
    *,
    video_path: Path,
    output_dir: Path,
    rollout_id: str,
) -> Path | None:
    keyframe_dir = output_dir / "wam_success_label_keyframes"
    ensure_dir(keyframe_dir)
    keyframe_path = keyframe_dir / f"{_safe_stem(rollout_id)}.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(keyframe_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not keyframe_path.is_file():
        return None
    return keyframe_path


def _gemini_label_one(
    *,
    client: Any,
    types_module: Any,
    model: str,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    video_path: Path,
    keyframe_path: Path | None = None,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    task_record = _rollout_task_prompt_record(request, rollout)
    task_prompt = _string(task_record.get("task_prompt"))
    success_criteria = _task_success_criteria(
        request=request,
        rollout=rollout,
        task_record=task_record,
    )
    prompt = {
        "instruction": (
            "You are judging a generated world-model rollout video for a robot manipulation task. "
            "The inputs include the MP4 and, when available, a still keyframe image extracted from that MP4. "
            "Use the visible robot, scene objects, and target evidence in either input. "
            "Return compact JSON only. Judge whether the generated video shows realistic task success. "
            "Be strict: do not infer success from provider completion, scene motion, camera motion, or a valid video. "
            "If the robot does not visibly reach/contact the correct target, or if the target state change is not "
            "visibly caused by the robot, mark success false or null."
        ),
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
        },
        "claim_boundary": (
            "This is only a semantic label on generated video, not physical robot proof."
        ),
    }
    contents: list[Any] = [json.dumps(prompt, sort_keys=True)]
    contents.append(types_module.Part.from_bytes(data=video_path.read_bytes(), mime_type=mime_type))
    if keyframe_path and keyframe_path.is_file():
        keyframe_mime = mimetypes.guess_type(keyframe_path.name)[0] or "image/jpeg"
        contents.append(
            types_module.Part.from_bytes(
                data=keyframe_path.read_bytes(),
                mime_type=keyframe_mime,
            )
        )
    try:
        config = types_module.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception:
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
    text = _string(getattr(response, "text", "")) or _string(getattr(response, "output_text", ""))
    payload = _parse_json_text(text or "{}")
    label_payload = payload
    if isinstance(payload.get("labels"), list) and payload["labels"]:
        first = payload["labels"][0]
        label_payload = dict(first) if isinstance(first, Mapping) else payload
    success = _bool_or_none(label_payload.get("success"))
    return {
        "label_id": f"gemini_{_string(rollout.get('rollout_id')) or 'rollout'}",
        "rollout_id": rollout.get("rollout_id"),
        "scenario_eval_run_id": rollout.get("scenario_eval_run_id"),
        "policy_id": rollout.get("policy_id"),
        "status": "success" if success is True else "failure" if success is False else "uncertain",
        "success": success,
        "confidence": _confidence_or_none(label_payload.get("confidence")),
        "rationale": _string(label_payload.get("rationale")) or None,
        "scene_description": _string(label_payload.get("scene_description")) or None,
        "task_completion_evidence": label_payload.get("task_completion_evidence")
        if isinstance(label_payload.get("task_completion_evidence"), list)
        else [],
        "failure_modes": label_payload.get("failure_modes")
        if isinstance(label_payload.get("failure_modes"), list)
        else [],
        "end_effector_reaches_target": _bool_or_none(
            label_payload.get("end_effector_reaches_target")
        ),
        "target_state_change_visible": _bool_or_none(
            label_payload.get("target_state_change_visible")
        ),
        "robot_caused_target_motion": _bool_or_none(
            label_payload.get("robot_caused_target_motion")
        ),
        "task_success_criteria": success_criteria,
        "evidence_refs": [
            str(path.resolve())
            for path in (video_path, keyframe_path)
            if path is not None and path.exists()
        ],
        "label_source": "gemini_generated_video_judge",
        "model": model,
        "visual_evidence_used": True,
        "keyframe_evidence_used": bool(keyframe_path and keyframe_path.is_file()),
        "human_review_required": False,
        "human_review_recommended": True,
        "rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "safety_or_contact_validation_proven": False,
        "srcc_or_policy_ranking_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def build_gemini_wam_success_labels(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    model: str | None = None,
    max_rollouts: int = 5,
    max_inline_bytes: int | None = None,
) -> dict[str, Any]:
    resolved_input = Path(input_path).resolve()
    resolved_output = Path(
        output_path
        or os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT")
        or resolved_input.parent / DEFAULT_OUTPUT_FILENAME
    ).resolve()
    ensure_dir(resolved_output.parent)
    output_dir = resolved_output.parent
    generated_at = utc_now_iso()
    request = _load_json(resolved_input)
    requested_model = _string(model or os.getenv(MODEL_ENV))
    model_name = requested_model if requested_model and "flash" in requested_model.lower() else DEFAULT_MODEL
    max_bytes = max_inline_bytes or int(
        _string(os.getenv("BLUEPRINT_GEMINI_WAM_SUCCESS_MAX_INLINE_BYTES"))
        or DEFAULT_MAX_INLINE_BYTES
    )
    blockers: list[str] = []
    labels: list[dict[str, Any]] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    api_key, api_key_source = _api_key()
    if not api_key:
        blockers.append("missing_gemini_google_genai_or_google_ai_api_key_or_key_file")
    rollouts = [
        dict(item)
        for item in request.get("rollouts", []) or []
        if isinstance(item, Mapping)
    ][:max_rollouts]
    if not rollouts:
        blockers.append("missing_generated_rollouts")

    if not blockers:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            blockers.append("missing_google_genai_package")
        else:
            client = genai.Client(api_key=api_key)
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
                video_text = _string(rollout.get("generated_video_path"))
                video_path = Path(video_text).expanduser()
                if not video_path.is_absolute():
                    video_path = resolved_input.parent / video_path
                if not video_path.is_file():
                    blockers.append("generated_video_path_not_found")
                    continue
                try:
                    size_bytes = video_path.stat().st_size
                except OSError:
                    blockers.append("generated_video_stat_failed")
                    continue
                if size_bytes > max_bytes:
                    blockers.append("generated_video_too_large_for_inline_gemini_label")
                    continue
                keyframe_path = _extract_first_keyframe(
                    video_path=video_path,
                    output_dir=output_dir,
                    rollout_id=_string(rollout.get("rollout_id")),
                )
                try:
                    labels.append(
                        _gemini_label_one(
                            client=client,
                            types_module=types,
                            model=model_name,
                            request=request,
                            rollout=rollout,
                            video_path=video_path,
                            keyframe_path=keyframe_path,
                        )
                    )
                except Exception as exc:  # pragma: no cover - live provider behavior
                    blockers.append(_provider_error_blocker(exc))
                    break

    manifest = {
        "schema_version": "wam_success_labels.command.v1",
        "generated_at": generated_at,
        "status": "completed" if labels and not blockers else "blocked",
        "provider": "gemini",
        "model": model_name,
        "api_key_configured": bool(api_key_source),
        "blockers": sorted(set(blockers)),
        "label_count": len(labels),
        "labels": labels,
        "visual_evidence_used": bool(labels),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "success_label_is_from_generated_video_not_physical_robot": True,
            "success_label_does_not_prove_forward_inverse_consistency": True,
            "generated_world_policy_evaluation_scope_proven": False,
        },
    }
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
    parser.add_argument("--output", type=Path, default=os.getenv("BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rollouts", type=int, default=5)
    args = parser.parse_args(argv)
    result = build_gemini_wam_success_labels(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_rollouts=args.max_rollouts,
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
