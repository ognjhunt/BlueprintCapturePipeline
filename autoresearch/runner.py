"""Search runner for skill-only autoresearch."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import (
    REPO_ROOT,
    copy_relative_files,
    ensure_dir,
    line_count,
    load_target_manifest,
    normalize_relpath,
    read_json,
    run_pytest,
    utc_now_compact,
    utc_now_iso,
    validate_target_manifest,
    write_json,
    write_text,
)
from .run_eval import evaluate_candidate
from .score import score_eval_payload, should_accept_candidate


DEFAULT_MAX_CHANGED_FILES = 3
DEFAULT_MAX_CHANGED_LINES = 120
DEFAULT_ITERATIONS = 20
DEFAULT_PATIENCE = 5


@dataclass(frozen=True)
class DiffSummary:
    changed_files: list[str]
    diff_text: str
    diff_size_lines: int


def _manifest_mutable_paths(manifest: Mapping[str, Any]) -> list[str]:
    return [normalize_relpath(item) for item in manifest.get("mutable_paths", [])] + [
        normalize_relpath(item) for item in manifest.get("optional_mutable_paths", [])
    ]


def snapshot_candidate_files(
    *,
    manifest: Mapping[str, Any],
    source_root: Path,
    destination_root: Path,
) -> None:
    copy_relative_files(
        source_root,
        destination_root,
        manifest.get("mutable_paths", []),
        allow_missing=False,
    )
    copy_relative_files(
        source_root,
        destination_root,
        manifest.get("optional_mutable_paths", []),
        allow_missing=True,
    )


def compute_candidate_diff(base_dir: Path, candidate_dir: Path) -> DiffSummary:
    changed_files: list[str] = []
    diff_chunks: list[str] = []
    all_paths = {
        normalize_relpath(str(path.relative_to(base_dir)))
        for path in base_dir.rglob("*")
        if path.is_file()
    } | {
        normalize_relpath(str(path.relative_to(candidate_dir)))
        for path in candidate_dir.rglob("*")
        if path.is_file()
    }
    for relative_path in sorted(all_paths):
        base_path = base_dir / relative_path
        candidate_path = candidate_dir / relative_path
        base_text = base_path.read_text(encoding="utf-8") if base_path.is_file() else ""
        candidate_text = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else ""
        if base_text == candidate_text:
            continue
        changed_files.append(relative_path)
        diff = difflib.unified_diff(
            base_text.splitlines(),
            candidate_text.splitlines(),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            lineterm="",
        )
        diff_chunks.append("\n".join(diff))
    diff_text = "\n".join(chunk for chunk in diff_chunks if chunk).strip()
    return DiffSummary(
        changed_files=changed_files,
        diff_text=diff_text + ("\n" if diff_text else ""),
        diff_size_lines=line_count(diff_text),
    )


def validate_diff_summary(
    diff_summary: DiffSummary,
    *,
    allowed_paths: set[str],
    max_changed_files: int,
    max_changed_lines: int,
) -> tuple[bool, list[str], bool]:
    reasons: list[str] = []
    forbidden = [path for path in diff_summary.changed_files if path not in allowed_paths]
    if forbidden:
        reasons.append(f"forbidden paths changed: {', '.join(sorted(forbidden))}")
    if not diff_summary.changed_files:
        reasons.append("no meaningful diff")
    if len(diff_summary.changed_files) > max_changed_files:
        reasons.append(
            f"changed file count {len(diff_summary.changed_files)} exceeds limit {max_changed_files}"
        )
    if diff_summary.diff_size_lines > max_changed_lines:
        reasons.append(
            f"diff size {diff_summary.diff_size_lines} exceeds limit {max_changed_lines}"
        )
    return not reasons, reasons, bool(forbidden)


def rank_candidate_records(records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            -float(item.get("total_score", 0.0)),
            int(item.get("diff_size_lines", 0)),
            int(item.get("accepted_iteration", 0)),
        ),
    )


def _mutation_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "change_summary": {"type": "string"},
        },
        "required": ["hypothesis", "change_summary"],
        "additionalProperties": False,
    }


def _build_mutation_prompt(
    *,
    program_text: str,
    manifest: Mapping[str, Any],
    candidate_dir: Path,
    iteration: int,
    previous_score: float,
) -> str:
    allowed_files = _manifest_mutable_paths(manifest)
    local_files: list[str] = []
    for relative_path in allowed_files:
        candidate_path = candidate_dir / relative_path
        if candidate_path.is_file():
            local_files.append(
                f"## {relative_path}\n\n{candidate_path.read_text(encoding='utf-8')}\n"
            )
    return (
        "You are making one bounded mutation for the BlueprintCapturePipeline autoresearch harness.\n"
        "Edit only files in the current working directory that correspond to the allowed mutable files.\n"
        "Do not create or edit harness files, tests, or adjacent skills.\n"
        "Prefer a small, surgical change that could improve the score.\n"
        "After editing, return only a JSON object with hypothesis and change_summary.\n\n"
        f"Iteration: {iteration}\n"
        f"Current best score: {previous_score}\n"
        f"Allowed relative files: {json.dumps(allowed_files, indent=2)}\n\n"
        f"{program_text}\n\n"
        "Current candidate files:\n\n"
        + "\n".join(local_files)
    )


def _invoke_mutation_agent(
    *,
    candidate_dir: Path,
    program_text: str,
    manifest: Mapping[str, Any],
    iteration: int,
    previous_score: float,
    mutation_engine: str,
    mutation_model: str,
    mutation_bin: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, str]:
    if mutation_engine == "noop":
        return {
            "hypothesis": "No mutation applied.",
            "change_summary": "Runner invoked noop mutation engine; no edits were made.",
        }
    if mutation_engine != "codex":
        raise ValueError(f"Unsupported mutation engine: {mutation_engine}")
    codex_path = shutil.which(mutation_bin)
    if not codex_path:
        raise RuntimeError(f"Could not find mutation engine binary: {mutation_bin}")

    with tempfile.TemporaryDirectory(prefix="autoresearch-mutate-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        schema_path = tmp_root / "schema.json"
        output_path = tmp_root / "proposal.json"
        schema_path.write_text(json.dumps(_mutation_output_schema(), indent=2), encoding="utf-8")
        command = [
            codex_path,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--cd",
            str(candidate_dir),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
        ]
        if mutation_model:
            command.extend(["--model", mutation_model])
        if reasoning_effort:
            command.extend(["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"])
        command.append("-")
        prompt = _build_mutation_prompt(
            program_text=program_text,
            manifest=manifest,
            candidate_dir=candidate_dir,
            iteration=iteration,
            previous_score=previous_score,
        )
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
                f"Mutation engine failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        payload = read_json(output_path)
        return {
            "hypothesis": str(payload.get("hypothesis") or "").strip(),
            "change_summary": str(payload.get("change_summary") or "").strip(),
        }


def run_preflight_repo_tests(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    summary = run_pytest(
        [str(item) for item in manifest.get("preflight_repo_tests", [])],
        cwd=repo_root,
    )
    payload = summary.to_dict()
    payload["generated_at"] = utc_now_iso()
    return payload


def _write_iteration_metadata(
    iteration_dir: Path,
    *,
    hypothesis: str,
    change_summary: str,
    rejection_reasons: list[str] | None = None,
) -> None:
    body = [
        f"Hypothesis: {hypothesis or 'N/A'}",
        "",
        "Change Summary:",
        change_summary or "N/A",
    ]
    if rejection_reasons:
        body.extend(["", "Rejection Reasons:", *[f"- {reason}" for reason in rejection_reasons]])
    write_text(iteration_dir / "proposal.md", "\n".join(body).strip() + "\n")


def _build_rejected_eval_payload(
    *,
    target_skill: str,
    iteration: int,
    rejection_reasons: list[str],
    preflight_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_skill": target_skill,
        "iteration": iteration,
        "generated_at": utc_now_iso(),
        "pytest": {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "exit_code": 0,
            "selected_tests": [],
            "pass_rate": 0.0,
            "stdout": "",
            "stderr": "",
            "junit_xml": "",
        },
        "structured_checks": {
            "passed_checks": 0,
            "total_checks": 0,
            "rate": 0.0,
            "required_fields_present": False,
            "schema_valid": False,
            "missing_fields": rejection_reasons,
        },
        "rubric": {
            "groundedness": 0.0,
            "completeness": 0.0,
            "conciseness": 0.0,
            "operator_utility": 0.0,
            "contract_preservation": 0.0,
            "score": 0.0,
        },
        "penalties": 0.0,
        "case_results": [],
        "preflight_repo_tests": dict(preflight_summary),
        "status": "rejected_pre_eval",
        "rejection_reasons": list(rejection_reasons),
    }


def _copy_ranked_candidates(run_dir: Path, accepted_records: list[Mapping[str, Any]]) -> None:
    ranked = rank_candidate_records(list(accepted_records))[:3]
    top_root = ensure_dir(run_dir / "top_candidates")
    for index, record in enumerate(ranked, start=1):
        destination = top_root / str(index)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(Path(str(record["candidate_dir"])), destination)


def _write_summary(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    baseline_score: Mapping[str, Any],
    best_score: Mapping[str, Any],
    accepted_records: list[Mapping[str, Any]],
    rejected_records: list[Mapping[str, Any]],
) -> None:
    final_diff = read_text = ""
    best_dir = run_dir / "best"
    if best_dir.exists():
        baseline_dir = run_dir / "baseline" / "candidate"
        final_diff = compute_candidate_diff(baseline_dir, best_dir).diff_text
    lines = [
        f"# Autoresearch Summary: {manifest['target_skill']}",
        "",
        f"- Target skill: `{manifest['target_skill']}`",
        f"- Baseline score: `{baseline_score['total_score']}`",
        f"- Best score: `{best_score['total_score']}`",
        f"- Accepted iterations: `{len(accepted_records)}`",
        f"- Rejected iterations: `{len(rejected_records)}`",
        "",
        "## Final Diff Summary",
    ]
    if final_diff:
        lines.extend(["```diff", final_diff.rstrip(), "```"])
    else:
        lines.append("No accepted mutations changed the baseline candidate.")
    lines.extend(
        [
            "",
            "## Residual Risks",
            "- Existing repo preflight tests do not currently consume mutated SKILL.md content directly.",
            "- Locked harness tests are the primary comparative signal in v1.",
            "- External agent availability remains a runtime dependency for live search runs.",
        ]
    )
    write_text(run_dir / "summary.md", "\n".join(lines) + "\n")


def run_search(
    *,
    target: str,
    iterations: int = DEFAULT_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
    mutation_engine: str = "codex",
    mutation_model: str = "gpt-5.4",
    mutation_bin: str = "codex",
    eval_agent_engine: str = "codex",
    eval_model: str = "gpt-5.4",
    eval_agent_bin: str = "codex",
    reasoning_effort: str = "medium",
    timeout_seconds: int = 120,
    max_changed_files: int = DEFAULT_MAX_CHANGED_FILES,
    max_changed_lines: int = DEFAULT_MAX_CHANGED_LINES,
) -> dict[str, Any]:
    manifest_path = REPO_ROOT / "autoresearch" / "targets" / f"{target}.json"
    manifest = load_target_manifest(manifest_path)
    validate_target_manifest(manifest)

    program_path = REPO_ROOT / "autoresearch" / "program.md"
    program_text = program_path.read_text(encoding="utf-8")
    run_dir = ensure_dir(REPO_ROOT / "autoresearch" / "runs" / target / utc_now_compact())
    write_text(run_dir / "program.md", program_text)
    write_json(run_dir / "target_manifest.json", dict(manifest))

    preflight_summary = run_preflight_repo_tests(manifest, repo_root=REPO_ROOT)
    write_json(run_dir / "preflight_repo_tests.json", preflight_summary)
    if preflight_summary["failed"] > 0 or preflight_summary["exit_code"] != 0:
        raise RuntimeError("Preflight repo tests failed; aborting autoresearch run.")

    baseline_candidate_dir = ensure_dir(run_dir / "baseline" / "candidate")
    snapshot_candidate_files(manifest=manifest, source_root=REPO_ROOT, destination_root=baseline_candidate_dir)
    baseline_eval_dir = ensure_dir(run_dir / "baseline" / "eval")
    baseline_eval = evaluate_candidate(
        target_manifest_path=manifest_path,
        candidate_dir=baseline_candidate_dir,
        output_dir=baseline_eval_dir,
        agent_engine=eval_agent_engine,
        model=eval_model,
        agent_bin=eval_agent_bin,
        reasoning_effort=reasoning_effort,
        timeout_seconds=timeout_seconds,
        iteration=0,
        preflight_summary=preflight_summary,
    )
    baseline_score = score_eval_payload(baseline_eval, diff_size_lines=0, forbidden_mutation_detected=False)
    baseline_score["accepted"] = True
    baseline_score["accept_reason"] = "baseline"
    write_json(run_dir / "baseline_eval.json", baseline_eval)
    write_json(run_dir / "baseline_score.json", baseline_score)

    best_candidate_dir = run_dir / "best"
    if best_candidate_dir.exists():
        shutil.rmtree(best_candidate_dir)
    shutil.copytree(baseline_candidate_dir, best_candidate_dir)
    best_score = dict(baseline_score)

    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    non_improving_rounds = 0
    allowed_paths = set(_manifest_mutable_paths(manifest))

    for iteration in range(1, max(0, int(iterations)) + 1):
        iteration_dir = ensure_dir(run_dir / "iterations" / str(iteration))
        candidate_dir = ensure_dir(iteration_dir / "candidate")
        snapshot_candidate_files(manifest=manifest, source_root=best_candidate_dir, destination_root=candidate_dir)
        proposal = _invoke_mutation_agent(
            candidate_dir=candidate_dir,
            program_text=program_text,
            manifest=manifest,
            iteration=iteration,
            previous_score=float(best_score["total_score"]),
            mutation_engine=mutation_engine,
            mutation_model=mutation_model,
            mutation_bin=mutation_bin,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )
        diff_summary = compute_candidate_diff(best_candidate_dir, candidate_dir)
        write_text(iteration_dir / "diff.patch", diff_summary.diff_text)
        diff_ok, rejection_reasons, forbidden_mutation_detected = validate_diff_summary(
            diff_summary,
            allowed_paths=allowed_paths,
            max_changed_files=max_changed_files,
            max_changed_lines=max_changed_lines,
        )
        _write_iteration_metadata(
            iteration_dir,
            hypothesis=proposal.get("hypothesis", ""),
            change_summary=proposal.get("change_summary", ""),
            rejection_reasons=None if diff_ok else rejection_reasons,
        )
        if not diff_ok:
            eval_payload = _build_rejected_eval_payload(
                target_skill=target,
                iteration=iteration,
                rejection_reasons=rejection_reasons,
                preflight_summary=preflight_summary,
            )
            score_payload = score_eval_payload(
                eval_payload,
                diff_size_lines=diff_summary.diff_size_lines,
                forbidden_mutation_detected=forbidden_mutation_detected,
            )
            score_payload["accepted"] = False
            score_payload["accept_reason"] = "rejected_pre_eval"
            write_json(iteration_dir / "eval.json", eval_payload)
            write_json(iteration_dir / "score.json", score_payload)
            rejected_records.append(
                {
                    **score_payload,
                    "iteration": iteration,
                    "candidate_dir": str(candidate_dir),
                }
            )
            continue

        eval_payload = evaluate_candidate(
            target_manifest_path=manifest_path,
            candidate_dir=candidate_dir,
            output_dir=iteration_dir,
            agent_engine=eval_agent_engine,
            model=eval_model,
            agent_bin=eval_agent_bin,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            iteration=iteration,
            preflight_summary=preflight_summary,
        )
        score_payload = score_eval_payload(
            eval_payload,
            diff_size_lines=diff_summary.diff_size_lines,
            forbidden_mutation_detected=forbidden_mutation_detected,
        )
        accepted, accept_reason = should_accept_candidate(
            best_score=float(best_score["total_score"]),
            best_diff_size_lines=int(best_score.get("diff_size_lines", 0)),
            candidate_score=float(score_payload["total_score"]),
            candidate_diff_size_lines=diff_summary.diff_size_lines,
        )
        score_payload["accepted"] = accepted
        score_payload["accept_reason"] = accept_reason
        write_json(iteration_dir / "score.json", score_payload)
        if accepted:
            if best_candidate_dir.exists():
                shutil.rmtree(best_candidate_dir)
            shutil.copytree(candidate_dir, best_candidate_dir)
            best_score = dict(score_payload)
            best_score["accepted_iteration"] = iteration
            accepted_records.append(
                {
                    **score_payload,
                    "accepted_iteration": iteration,
                    "candidate_dir": str(candidate_dir),
                }
            )
            non_improving_rounds = 0
        else:
            rejected_records.append(
                {
                    **score_payload,
                    "iteration": iteration,
                    "candidate_dir": str(candidate_dir),
                }
            )
            if accept_reason.startswith("tie_") or accept_reason == "lower_score":
                non_improving_rounds += 1
        if non_improving_rounds >= max(0, int(patience)):
            break

    final_preflight_summary = run_preflight_repo_tests(manifest, repo_root=REPO_ROOT)
    write_json(run_dir / "final_preflight_repo_tests.json", final_preflight_summary)
    _copy_ranked_candidates(run_dir, accepted_records)
    _write_summary(
        run_dir=run_dir,
        manifest=manifest,
        baseline_score=baseline_score,
        best_score=best_score,
        accepted_records=accepted_records,
        rejected_records=rejected_records,
    )
    return {
        "run_dir": str(run_dir),
        "baseline_score": baseline_score,
        "best_score": best_score,
        "accepted_iterations": len(accepted_records),
        "rejected_iterations": len(rejected_records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Blueprint autoresearch search loop")
    parser.add_argument("--target", required=True, choices=("intake_normalizer", "readiness_report_writer", "recapture_planner"))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--mutation-engine", default="codex")
    parser.add_argument("--mutation-model", default="gpt-5.4")
    parser.add_argument("--mutation-bin", default="codex")
    parser.add_argument("--eval-agent-engine", default="codex")
    parser.add_argument("--eval-model", default="gpt-5.4")
    parser.add_argument("--eval-agent-bin", default="codex")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-changed-files", type=int, default=DEFAULT_MAX_CHANGED_FILES)
    parser.add_argument("--max-changed-lines", type=int, default=DEFAULT_MAX_CHANGED_LINES)
    args = parser.parse_args(argv)

    run_search(
        target=args.target,
        iterations=args.iterations,
        patience=args.patience,
        mutation_engine=args.mutation_engine,
        mutation_model=args.mutation_model,
        mutation_bin=args.mutation_bin,
        eval_agent_engine=args.eval_agent_engine,
        eval_model=args.eval_model,
        eval_agent_bin=args.eval_agent_bin,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        max_changed_files=args.max_changed_files,
        max_changed_lines=args.max_changed_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
