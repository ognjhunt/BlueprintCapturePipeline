"""Gemini-backed episode consistency labels for WAM-generated rollout videos.

The command consumes ``wam_episode_consistency_request.json`` and writes
``wam_episode_consistency.command.json``. Labels are external judgments over a
generated video plus trace context; they do not run the WAM and do not upgrade
task success, safety, deployment, SRCC, or generated-world rank fidelity claims.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


GATE_ENV = "BLUEPRINT_ALLOW_GEMINI_WAM_EPISODE_CONSISTENCY"
MODEL_ENV = "BLUEPRINT_GEMINI_WAM_EPISODE_CONSISTENCY_MODEL"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OUTPUT_FILENAME = "wam_episode_consistency.command.json"
DEFAULT_MAX_INLINE_BYTES = 95 * 1024 * 1024


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
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "prepayment credits are depleted" in text:
        return "gemini_resource_exhausted_or_billing_credits_depleted"
    if "PERMISSION_DENIED" in text:
        return "gemini_permission_denied"
    if "INVALID_ARGUMENT" in text:
        return "gemini_invalid_argument"
    return f"gemini_wam_episode_consistency_failed:{type(exc).__name__}"


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


def _gemini_score_one(
    *,
    client: Any,
    types_module: Any,
    model: str,
    request: Mapping[str, Any],
    rollout: Mapping[str, Any],
    video_path: Path,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    prompt = {
        "instruction": (
            "You are an external evaluator, not the WAM. Judge whether a generated "
            "world-model rollout is forward/inverse consistent with the provided "
            "task prompt and trace summary. Return compact JSON only. Do not judge "
            "generated-world rank fidelity or deployment safety."
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
        },
        "trace_summary": request.get("trace_summary"),
        "claim_boundary": (
            "This is an external consistency label on generated video and trace context only."
        ),
    }
    contents: list[Any] = [json.dumps(prompt, sort_keys=True)]
    contents.append(types_module.Part.from_bytes(data=video_path.read_bytes(), mime_type=mime_type))
    try:
        config = types_module.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception:
        response = client.models.generate_content(model=model, contents=contents)
    text = _string(getattr(response, "text", "")) or _string(getattr(response, "output_text", ""))
    payload = _parse_json_text(text or "{}")
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
        "evidence_refs": [str(video_path.resolve())],
        "label_source": "gemini_wam_episode_consistency_judge",
        "model": model,
        "visual_evidence_used": True,
        "action_trace_evidence_used": True,
        **_consistency_support_claim_boundary(),
        "task_success_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "safety_or_contact_validation_proven": False,
        "srcc_or_policy_ranking_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def build_gemini_wam_episode_consistency_labels(
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
        or os.getenv("BLUEPRINT_WAM_CONSISTENCY_OUTPUT")
        or resolved_input.parent / DEFAULT_OUTPUT_FILENAME
    ).resolve()
    ensure_dir(resolved_output.parent)
    request = _load_json(resolved_input)
    requested_model = _string(model or os.getenv(MODEL_ENV))
    model_name = requested_model if requested_model and "flash" in requested_model.lower() else DEFAULT_MODEL
    max_bytes = max_inline_bytes or int(
        _string(os.getenv("BLUEPRINT_GEMINI_WAM_EPISODE_CONSISTENCY_MAX_INLINE_BYTES"))
        or DEFAULT_MAX_INLINE_BYTES
    )
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
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
                    blockers.append("generated_video_too_large_for_inline_gemini_consistency")
                    continue
                try:
                    checks.append(
                        _gemini_score_one(
                            client=client,
                            types_module=types,
                            model=model_name,
                            request=request,
                            rollout=rollout,
                            video_path=video_path,
                        )
                    )
                except Exception as exc:  # pragma: no cover - live provider behavior
                    blockers.append(_provider_error_blocker(exc))
                    break

    manifest = {
        "schema_version": "wam_episode_consistency.command.v1",
        "generated_at": utc_now_iso(),
        "status": "completed" if checks and not blockers else "blocked",
        "provider": "gemini_wam_episode_consistency_judge",
        "model": model_name,
        "api_key_configured": bool(api_key_source),
        "blockers": sorted(set(blockers)),
        "rollout_check_count": len(checks),
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
    args = parser.parse_args(argv)
    result = build_gemini_wam_episode_consistency_labels(
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
                "rollout_check_count": result["rollout_check_count"],
                "blockers": result["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
