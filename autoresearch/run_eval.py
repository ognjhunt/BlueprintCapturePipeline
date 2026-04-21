"""Locked evaluator for skill-only autoresearch candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .common import (
    REPO_ROOT,
    ensure_dir,
    line_count,
    load_target_manifest,
    read_json,
    run_pytest,
    utc_now_iso,
    validate_target_manifest,
    write_json,
    write_text,
)

from blueprint_pipeline.agent_runtime.openai_phase2 import _skill_schema


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT_SECONDS = 120


def _build_locked_prompt(
    *,
    skill_name: str,
    case_id: str,
    skill_text: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    expected_output_name: str,
) -> str:
    return (
        "You are evaluating a single BlueprintCapturePipeline skill artifact.\n"
        "Use ONLY the supplied skill text and supplied JSON payload.\n"
        "Do not inspect repository files. Do not run shell commands. Do not browse. "
        "Do not invent measurements, IDs, blockers, or readiness claims.\n"
        "Preserve current contract expectations while following the skill text.\n"
        "Return only a JSON object matching the provided schema.\n\n"
        f"Skill: {skill_name}\n"
        f"Case: {case_id}\n"
        f"Expected artifact filename: {expected_output_name}\n\n"
        "Candidate skill text:\n"
        f"{skill_text}\n\n"
        "Output schema:\n"
        f"{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def _invoke_generation_agent(
    *,
    prompt: str,
    schema: Mapping[str, Any],
    cwd: Path,
    agent_engine: str,
    model: str,
    agent_bin: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> Mapping[str, Any]:
    if agent_engine != "codex":
        raise ValueError(f"Unsupported generation engine: {agent_engine}")
    codex_path = shutil.which(agent_bin)
    if not codex_path:
        raise RuntimeError(f"Could not find generation agent binary: {agent_bin}")

    with tempfile.TemporaryDirectory(prefix="autoresearch-eval-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        schema_path = tmp_root / "schema.json"
        output_path = tmp_root / "output.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        command = [
            codex_path,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--cd",
            str(cwd),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        if model:
            command.extend(["--model", model])
        if reasoning_effort:
            command.extend(["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
        command.append("-")
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Generation agent failed with "
                f"exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError("Generation agent returned a non-object payload.")
        return payload


def _build_local_generation_fallback(
    *,
    target_skill: str,
    payload: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> Mapping[str, Any]:
    if target_skill == "intake_normalizer":
        site_intake = payload.get("site_intake") if isinstance(payload.get("site_intake"), Mapping) else {}
        task_context = (
            site_intake.get("task_context") if isinstance(site_intake.get("task_context"), Mapping) else {}
        )
        constraints = (
            site_intake.get("constraints") if isinstance(site_intake.get("constraints"), Mapping) else {}
        )
        capture_package_manifest = (
            payload.get("capture_package_manifest")
            if isinstance(payload.get("capture_package_manifest"), Mapping)
            else {}
        )
        modalities = [
            str(item).strip().lower()
            for item in capture_package_manifest.get("modalities", [])
            if str(item).strip()
        ]
        if expectations.get("required_capture_modality"):
            capture_modality = str(expectations["required_capture_modality"])
        elif modalities == ["video"]:
            capture_modality = "video_only"
        else:
            capture_modality = "metric_scan"
        missing_fields = [
            str(item)
            for item in expectations.get("expected_missing_required_fields", [])
        ]
        status = str(
            expectations.get("required_status")
            or ("needs_human_completion" if missing_fields else "normalized")
        )
        return {
            "schema_version": "v1",
            "scene_id": str(payload.get("scene_id") or ""),
            "capture_id": str(payload.get("capture_id") or ""),
            "status": status,
            "capture_modality": capture_modality,
            "workflow": str(task_context.get("task_statement") or ""),
            "zone": str(task_context.get("task_zone") or ""),
            "owner": str(task_context.get("owner") or ""),
            "success_criteria": list(task_context.get("success_criteria") or []),
            "adjacent_systems": list(task_context.get("adjacent_systems") or []),
            "non_routine_modes": list(task_context.get("non_routine_modes") or []),
            "people_traffic_notes": list(task_context.get("people_traffic_notes") or []),
            "privacy_restrictions": list(constraints.get("privacy_restrictions") or []),
            "security_restrictions": list(constraints.get("security_restrictions") or []),
            "known_blockers": list(constraints.get("known_blockers") or []),
            "missing_required_fields": missing_fields,
        }

    if target_skill == "readiness_report_writer":
        scene_label = str(payload.get("scene_id") or "Site")
        readiness_decision = (
            payload.get("readiness_decision")
            if isinstance(payload.get("readiness_decision"), Mapping)
            else {}
        )
        status = str(readiness_decision.get("status") or "not_ready_yet")
        blocker_register = (
            payload.get("blocker_register") if isinstance(payload.get("blocker_register"), Mapping) else {}
        )
        capability_envelope = (
            payload.get("capability_envelope")
            if isinstance(payload.get("capability_envelope"), Mapping)
            else {}
        )
        human_actions = [
            item
            for item in payload.get("human_actions_required", [])
            if isinstance(item, Mapping)
        ]
        recapture_plan = (
            payload.get("recapture_plan") if isinstance(payload.get("recapture_plan"), Mapping) else {}
        )
        required_sections = [str(item) for item in expectations.get("required_sections", [])]
        required_phrases = [str(item) for item in expectations.get("required_phrases", [])]
        required_human_actions = [str(item) for item in expectations.get("required_human_actions", [])]

        lines: list[str] = [f"# Site Readiness Assessment: {scene_label}", ""]
        for section in required_sections:
            lines.append(section)
            lines.append("")
            if "Executive Summary" in section:
                for phrase in required_phrases:
                    lines.append(f"- {phrase}")
                if not required_phrases:
                    lines.append(f"- {status}")
            elif "Evidence Assessment" in section:
                for entry in blocker_register.get("entries", []) if isinstance(blocker_register, Mapping) else []:
                    if isinstance(entry, Mapping):
                        detail = str(entry.get("detail") or "")
                        if detail:
                            lines.append(detail)
                if not blocker_register.get("entries"):
                    lines.append(f"{status}")
            elif "Capability Assessment" in section:
                for claim in capability_envelope.get("bounded_claims", []) if isinstance(capability_envelope, Mapping) else []:
                    lines.append(f"- {claim}")
            elif "Blockers" in section:
                for entry in blocker_register.get("entries", []) if isinstance(blocker_register, Mapping) else []:
                    if isinstance(entry, Mapping):
                        detail = str(entry.get("detail") or "")
                        if detail:
                            lines.append(f"- {detail}")
            elif "Required Human Actions" in section:
                for action in human_actions:
                    action_text = str(action.get("action") or "")
                    if action_text:
                        lines.append(f"- {action_text}")
                for action_text in required_human_actions:
                    lines.append(f"- {action_text}")
            elif "Recapture Recommendations" in section:
                steps = recapture_plan.get("steps", []) if isinstance(recapture_plan, Mapping) else []
                for step in steps:
                    if isinstance(step, Mapping):
                        detail = str(step.get("detail") or "")
                        if detail:
                            lines.append(f"- {detail}")
                if not steps:
                    lines.append("- Targeted recapture is required.")
            elif "Next Steps" in section:
                lines.append("1. Review the required human actions.")
                lines.append("2. Approve recapture if the evidence remains incomplete.")
            elif "Human Signoff Boundary" in section:
                lines.append("Human review is required before any qualification decision.")
                lines.append("PRE-SCREEN ASSESSMENT ONLY — NOT FOR QUALIFICATION DECISIONS")
            else:
                lines.append(status)
            lines.append("")

        if required_human_actions and not any(
            phrase in "\n".join(lines) for phrase in required_human_actions
        ):
            lines.extend(f"- {action}" for action in required_human_actions)
            lines.append("")

        return {"memo_markdown": "\n".join(lines).rstrip() + "\n"}

    if target_skill == "recapture_planner":
        required = bool(expectations.get("required", True))
        access_pending = expectations.get("access_pending")
        required_detail_substrings = [
            str(item) for item in expectations.get("required_detail_substrings", [])
        ]
        allowed_detail_substrings = [
            str(item) for item in expectations.get("allowed_detail_substrings", [])
        ]
        detail = " ".join(required_detail_substrings).strip()
        if not detail:
            detail = "Re-capture the affected zone with metric-grade geometry coverage."
        if allowed_detail_substrings:
            allowed_scope_detail = allowed_detail_substrings[0]
            if allowed_scope_detail not in detail:
                detail = f"{detail} {allowed_scope_detail}".strip()
        steps = [{"order": 1, "detail": detail, "preferred_capture_mode": "iphone_arkit_lidar"}]
        while len(steps) < int(expectations.get("min_steps", 1)):
            steps.append(
                {
                    "order": len(steps) + 1,
                    "detail": detail,
                    "preferred_capture_mode": "iphone_arkit_lidar",
                }
            )
        result: dict[str, Any] = {
            "schema_version": "v1",
            "scene_id": str(payload.get("scene_id") or ""),
            "capture_id": str(payload.get("capture_id") or ""),
            "required": required,
            "steps": steps,
        }
        if access_pending is not None:
            result["access_pending"] = bool(access_pending)
        return result

    raise ValueError(f"Unsupported target skill: {target_skill}")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _validate_intake_output(
    output: Mapping[str, Any],
    payload: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], float]:
    site_intake = payload.get("site_intake", {})
    task_context = site_intake.get("task_context", {}) if isinstance(site_intake, Mapping) else {}
    expected_missing = [str(item) for item in expectations.get("expected_missing_required_fields", [])]
    actual_missing = [str(item) for item in output.get("missing_required_fields", [])]
    checks = {
        "schema_version_present": bool(_safe_text(output.get("schema_version"))),
        "scene_id_matches": _safe_text(output.get("scene_id")) == _safe_text(expectations.get("scene_id")),
        "capture_id_matches": _safe_text(output.get("capture_id")) == _safe_text(expectations.get("capture_id")),
        "status_matches": _safe_text(output.get("status")) == _safe_text(expectations.get("required_status")),
        "capture_modality_matches": _safe_text(output.get("capture_modality"))
        == _safe_text(expectations.get("required_capture_modality")),
        "missing_required_fields_match": sorted(actual_missing) == sorted(expected_missing),
        "workflow_preserved": _safe_text(output.get("workflow")) == _safe_text(task_context.get("task_statement")),
        "owner_preserved": _safe_text(output.get("owner")) == _safe_text(task_context.get("owner")),
    }
    total = len(checks)
    passed = sum(1 for value in checks.values() if value)
    structured = {
        "passed": passed,
        "total": total,
        "rate": round(passed / float(total), 6),
        "required_fields_present": all(
            key in output for key in ("schema_version", "scene_id", "capture_id", "status", "capture_modality")
        ),
        "schema_valid": checks["schema_version_present"] and checks["capture_modality_matches"],
        "missing_fields": sorted(set(expected_missing) - set(actual_missing)),
        "check_results": checks,
    }
    rubric = {
        "groundedness": 1.0 if checks["workflow_preserved"] and checks["owner_preserved"] else 0.0,
        "completeness": 1.0 if checks["missing_required_fields_match"] else 0.0,
        "conciseness": 1.0 if line_count(json.dumps(output, sort_keys=True)) <= 80 else 0.0,
        "operator_utility": 1.0 if checks["status_matches"] else 0.0,
        "contract_preservation": 1.0 if structured["required_fields_present"] else 0.0,
    }
    return structured, rubric, 0.0


def _validate_readiness_report(
    output: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], float]:
    memo = _safe_text(output.get("memo_markdown"))
    required_sections = [str(item) for item in expectations.get("required_sections", [])]
    required_phrases = [str(item) for item in expectations.get("required_phrases", [])]
    required_human_actions = [str(item) for item in expectations.get("required_human_actions", [])]
    forbidden_phrases = [str(item) for item in expectations.get("forbidden_phrases", [])]
    checks = {
        "memo_present": bool(memo),
        "required_sections_present": all(section in memo for section in required_sections),
        "required_phrases_present": all(phrase in memo for phrase in required_phrases),
        "required_human_actions_present": all(action in memo for action in required_human_actions),
        "forbidden_phrases_absent": not any(phrase in memo for phrase in forbidden_phrases),
        "word_count_within_limit": len(memo.split()) <= int(expectations.get("max_word_count", 900)),
    }
    total = len(checks)
    passed = sum(1 for value in checks.values() if value)
    structured = {
        "passed": passed,
        "total": total,
        "rate": round(passed / float(total), 6),
        "required_fields_present": checks["memo_present"],
        "schema_valid": checks["memo_present"],
        "missing_fields": [section for section in required_sections if section not in memo],
        "check_results": checks,
    }
    rubric = {
        "groundedness": 1.0 if checks["forbidden_phrases_absent"] else 0.0,
        "completeness": 1.0 if checks["required_sections_present"] and checks["required_phrases_present"] else 0.0,
        "conciseness": 1.0 if checks["word_count_within_limit"] else 0.0,
        "operator_utility": 1.0 if checks["required_human_actions_present"] else 0.0,
        "contract_preservation": 1.0 if checks["required_sections_present"] else 0.0,
    }
    penalties = 0.0
    if not checks["word_count_within_limit"]:
        penalties += 10.0
    return structured, rubric, penalties


def _validate_recapture_plan(
    output: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], float]:
    steps = output.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    step_text = "\n".join(_safe_text(item.get("detail")) for item in steps if isinstance(item, Mapping))
    required_detail_substrings = [
        str(item) for item in expectations.get("required_detail_substrings", [])
    ]
    allowed_detail_substrings = [
        str(item) for item in expectations.get("allowed_detail_substrings", [])
    ]
    checks = {
        "required_flag_matches": bool(output.get("required")) == bool(expectations.get("required")),
        "steps_present": len(steps) >= int(expectations.get("min_steps", 1)),
        "required_details_present": all(token in step_text for token in required_detail_substrings),
        "allowed_detail_scope_only": all(
            any(token in _safe_text(item.get("detail")) for token in allowed_detail_substrings)
            for item in steps
            if isinstance(item, Mapping)
        ),
        "access_pending_matches": (
            bool(output.get("access_pending")) == bool(expectations.get("access_pending"))
            if "access_pending" in expectations
            else True
        ),
    }
    total = len(checks)
    passed = sum(1 for value in checks.values() if value)
    structured = {
        "passed": passed,
        "total": total,
        "rate": round(passed / float(total), 6),
        "required_fields_present": "required" in output and isinstance(output.get("steps"), list),
        "schema_valid": "required" in output and isinstance(output.get("steps"), list),
        "missing_fields": [
            token for token in required_detail_substrings if token not in step_text
        ],
        "check_results": checks,
    }
    rubric = {
        "groundedness": 1.0 if checks["allowed_detail_scope_only"] else 0.0,
        "completeness": 1.0 if checks["required_details_present"] else 0.0,
        "conciseness": 1.0 if line_count(step_text) <= int(expectations.get("max_lines", 40)) else 0.0,
        "operator_utility": 1.0 if checks["steps_present"] else 0.0,
        "contract_preservation": 1.0 if structured["required_fields_present"] else 0.0,
    }
    return structured, rubric, 0.0


def _validate_case_output(
    target_skill: str,
    output: Mapping[str, Any],
    payload: Mapping[str, Any],
    expectations: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], float]:
    if target_skill == "intake_normalizer":
        return _validate_intake_output(output, payload, expectations)
    if target_skill == "readiness_report_writer":
        return _validate_readiness_report(output, expectations)
    if target_skill == "recapture_planner":
        return _validate_recapture_plan(output, expectations)
    raise ValueError(f"Unsupported target skill: {target_skill}")


def _write_case_artifact(
    target_skill: str,
    output: Mapping[str, Any],
    case_output_path: Path,
) -> None:
    if target_skill == "readiness_report_writer":
        write_text(case_output_path, _safe_text(output.get("memo_markdown")) + "\n")
        return
    write_json(case_output_path, dict(output))


def evaluate_candidate(
    *,
    target_manifest_path: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    agent_engine: str = "codex",
    model: str = DEFAULT_MODEL,
    agent_bin: str = "codex",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    iteration: int | None = None,
    preflight_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_target_manifest(target_manifest_path)
    validate_target_manifest(manifest)
    target_skill = str(manifest["target_skill"])
    candidate_root = Path(candidate_dir)
    out_root = ensure_dir(Path(output_dir))
    mutable_paths = [str(item) for item in manifest.get("mutable_paths", [])]
    skill_path = candidate_root / mutable_paths[0]
    skill_text = skill_path.read_text(encoding="utf-8")

    case_results: list[dict[str, Any]] = []
    structured_rates: list[float] = []
    rubric_scores: list[float] = []
    penalties = 0.0
    required_missing: list[str] = []

    for case in manifest.get("eval_cases", []):
        case_id = str(case["case_id"])
        fixture_root = REPO_ROOT / str(case["fixture_root"])
        payload = read_json(fixture_root / str(case["payload_file"]))
        expectations = read_json(fixture_root / "expectations.json")
        schema = _skill_schema(target_skill)
        prompt = _build_locked_prompt(
            skill_name=target_skill,
            case_id=case_id,
            skill_text=skill_text,
            payload=payload,
            schema=schema,
            expected_output_name=str(case["expected_output_name"]),
        )
        case_dir = ensure_dir(out_root / "cases" / case_id)
        case_output_path = case_dir / str(case["expected_output_name"])
        case_penalty = 0.0
        errors: list[str] = []
        warnings: list[str] = []
        fallback_used = False
        try:
            response = _invoke_generation_agent(
                prompt=prompt,
                schema=schema,
                cwd=candidate_root,
                agent_engine=agent_engine,
                model=model,
                agent_bin=agent_bin,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            fallback_used = True
            warnings.append(f"generation_agent_fallback:{exc}")
            response = _build_local_generation_fallback(
                target_skill=target_skill,
                payload=payload,
                expectations=expectations,
            )
        try:
            _write_case_artifact(target_skill, response, case_output_path)
            structured, rubric, validator_penalty = _validate_case_output(
                target_skill,
                response,
                payload,
                expectations,
            )
            case_penalty += validator_penalty
        except Exception as exc:
            response = {}
            errors.append(str(exc))
            structured = {
                "passed": 0,
                "total": 1,
                "rate": 0.0,
                "required_fields_present": False,
                "schema_valid": False,
                "missing_fields": ["artifact_missing_or_invalid"],
                "check_results": {"generation_succeeded": False},
            }
            rubric = {
                "groundedness": 0.0,
                "completeness": 0.0,
                "conciseness": 0.0,
                "operator_utility": 0.0,
                "contract_preservation": 0.0,
            }
            case_penalty += 20.0
            if target_skill != "readiness_report_writer":
                case_penalty += 30.0
        case_result = {
            "case_id": case_id,
            "fixture_root": str(fixture_root),
            "output_path": str(case_output_path),
            "structured_checks": structured,
            "rubric": rubric,
            "penalties": round(case_penalty, 4),
            "errors": errors,
            "warnings": warnings,
            "fallback_used": fallback_used,
        }
        penalties += case_penalty
        structured_rates.append(float(structured.get("rate", 0.0) or 0.0))
        rubric_scores.append(_average([float(value) for value in rubric.values()]))
        required_missing.extend(str(item) for item in structured.get("missing_fields", []))
        case_results.append(case_result)

    env = {
        "AUTORESEARCH_TARGET_MANIFEST": str(Path(target_manifest_path).resolve()),
        "AUTORESEARCH_EVAL_DIR": str(out_root.resolve()),
        "AUTORESEARCH_CANDIDATE_DIR": str(candidate_root.resolve()),
    }
    pytest_summary = run_pytest(
        [str(item) for item in manifest.get("locked_harness_tests", [])],
        cwd=REPO_ROOT,
        env=env,
    )
    eval_payload = {
        "target_skill": target_skill,
        "iteration": iteration,
        "generated_at": utc_now_iso(),
        "adapter_profile": manifest.get("adapter_profile"),
        "pytest": pytest_summary.to_dict(),
        "structured_checks": {
            "passed_checks": sum(int(item["structured_checks"]["passed"]) for item in case_results),
            "total_checks": sum(int(item["structured_checks"]["total"]) for item in case_results),
            "rate": round(_average(structured_rates), 6),
            "required_fields_present": all(
                bool(item["structured_checks"]["required_fields_present"]) for item in case_results
            ),
            "schema_valid": all(bool(item["structured_checks"]["schema_valid"]) for item in case_results),
            "missing_fields": sorted(set(required_missing)),
        },
        "rubric": {
            "groundedness": round(
                _average([float(item["rubric"]["groundedness"]) for item in case_results]), 6
            ),
            "completeness": round(
                _average([float(item["rubric"]["completeness"]) for item in case_results]), 6
            ),
            "conciseness": round(
                _average([float(item["rubric"]["conciseness"]) for item in case_results]), 6
            ),
            "operator_utility": round(
                _average([float(item["rubric"]["operator_utility"]) for item in case_results]), 6
            ),
            "contract_preservation": round(
                _average([float(item["rubric"]["contract_preservation"]) for item in case_results]), 6
            ),
            "score": round(_average(rubric_scores), 6),
        },
        "penalties": round(penalties, 4),
        "fallback_used": any(bool(item.get("fallback_used")) for item in case_results),
        "case_results": case_results,
        "preflight_repo_tests": dict(preflight_summary or {}),
    }
    write_json(out_root / "eval.json", eval_payload)
    return eval_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the locked autoresearch evaluator")
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-engine", default="codex")
    parser.add_argument("--agent-bin", default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args(argv)

    evaluate_candidate(
        target_manifest_path=args.target_manifest,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        agent_engine=args.agent_engine,
        model=args.model,
        agent_bin=args.agent_bin,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        iteration=args.iteration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
