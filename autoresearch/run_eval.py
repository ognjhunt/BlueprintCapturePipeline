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
