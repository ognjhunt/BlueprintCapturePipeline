#!/usr/bin/env python3
"""Select and run the smallest deterministic PR test set.

The selector is intentionally conservative. Directly related tests and a small
always-on sentinel set run for ordinary changes. Dependency-wide, unknown, or
overly broad changes request the separately governed full lane instead of
silently pretending that a narrow selection is sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = 120
MAX_CHANGED_FILES = 50
MAX_IMPACTED_TEST_FILES = 40

SENTINEL_TESTS = (
    "tests/test_success_claim_contracts.py::test_freshness_fails_closed_without_any_signal",
    "tests/test_runtime_security_controls.py::test_bounded_service_url_allows_only_loopback_plain_http",
    "tests/test_paid_resource_admission.py::test_shared_chokepoint_grants_only_exact_admitted_contract",
    "tests/test_paid_resource_allocator_verifier.py::test_unmanifested_script_mutator_is_rejected",
    "tests/test_release_engineering_contracts.py::test_risk_based_verification_workflows_are_bounded",
)

CROSS_CUTTING_FILES = {
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "requirements-geometry.txt",
    "tests/conftest.py",
    "src/blueprint_pipeline/__init__.py",
    "Dockerfile",
    "docker-compose.yml",
}

POLICY_ONLY_PREFIXES = (
    ".github/",
    "docs/",
)

POLICY_ONLY_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "scripts/pytest_fast.sh",
    "scripts/pytest_full.sh",
    "src/blueprint_pipeline/impacted_test_selection.py",
}


def _normalize_paths(paths: Iterable[str]) -> list[str]:
    normalized = {
        Path(path.strip()).as_posix().removeprefix("./")
        for path in paths
        if path.strip()
    }
    return sorted(normalized)


def _git_lines(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.splitlines()


def changed_files(
    root: Path,
    *,
    base: str,
    head: str,
    include_worktree: bool,
) -> list[str]:
    if not base or set(base) == {"0"}:
        base = f"{head}^"
    paths = _git_lines(root, "diff", "--name-only", "--diff-filter=ACMR", f"{base}...{head}")
    if include_worktree:
        paths.extend(_git_lines(root, "diff", "--name-only", "--diff-filter=ACMR"))
        paths.extend(_git_lines(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR"))
        paths.extend(_git_lines(root, "ls-files", "--others", "--exclude-standard"))
    return _normalize_paths(paths)


def _test_sources(root: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in sorted((root / "tests").glob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        sources[relative] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def _matching_tests(
    *,
    changed_path: str,
    test_sources: dict[str, str],
) -> set[str]:
    path = Path(changed_path)
    stem = path.stem
    candidates: set[str] = set()

    direct = f"tests/test_{stem}.py"
    if direct in test_sources:
        candidates.add(direct)

    tokens = {changed_path, path.name}
    if changed_path.startswith("src/blueprint_pipeline/") and path.suffix == ".py":
        module = changed_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        tokens.add(module)
    elif changed_path.startswith("scripts/"):
        tokens.add(f"scripts/{path.name}")
    elif changed_path.startswith("tests/fixtures/"):
        tokens.add(path.parent.name)

    for test_path, source in test_sources.items():
        if test_path == "tests/test_impacted_test_selection.py":
            continue
        if any(token and token in source for token in tokens):
            candidates.add(test_path)
    return candidates


def build_plan(root: Path, paths: Sequence[str]) -> dict[str, object]:
    changed = _normalize_paths(paths)
    test_sources = _test_sources(root)
    impacted: set[str] = set()
    reasons: list[str] = []
    requires_full_suite = False

    if not changed:
        reasons.append("no_changed_files:sentinels_only")

    if len(changed) > MAX_CHANGED_FILES:
        requires_full_suite = True
        reasons.append(f"changed_file_count_exceeds_budget:{len(changed)}>{MAX_CHANGED_FILES}")

    for changed_path in changed:
        if changed_path in CROSS_CUTTING_FILES:
            requires_full_suite = True
            reasons.append(f"cross_cutting_file:{changed_path}")
            continue

        if changed_path.startswith("tests/") and Path(changed_path).name.startswith("test_"):
            impacted.add(changed_path)
            reasons.append(f"changed_test:{changed_path}")
            continue

        if changed_path in POLICY_ONLY_FILES or changed_path.startswith(POLICY_ONLY_PREFIXES):
            impacted.update(
                _matching_tests(changed_path=changed_path, test_sources=test_sources)
            )
            if changed_path.startswith(".github/") or "CI_" in changed_path:
                impacted.add("tests/test_release_engineering_contracts.py")
            reasons.append(f"policy_or_documentation:{changed_path}")
            continue

        matches = _matching_tests(changed_path=changed_path, test_sources=test_sources)
        if matches:
            impacted.update(matches)
            reasons.append(f"mapped:{changed_path}:{len(matches)}")
            continue

        if changed_path.startswith(("src/", "scripts/", "deploy/", "terraform/")):
            requires_full_suite = True
            reasons.append(f"unmapped_executable_surface:{changed_path}")
        else:
            reasons.append(f"non_executable_surface:{changed_path}")

    if len(impacted) > MAX_IMPACTED_TEST_FILES:
        requires_full_suite = True
        reasons.append(
            f"impacted_test_count_exceeds_budget:{len(impacted)}>{MAX_IMPACTED_TEST_FILES}"
        )
        impacted.clear()

    selected_set = set(impacted)
    selected_set.update(
        sentinel
        for sentinel in SENTINEL_TESTS
        if sentinel.split("::", 1)[0] not in impacted
    )
    selected = sorted(selected_set)
    return {
        "schema_version": "blueprint_impacted_test_plan.v1",
        "changed_files": changed,
        "selected_tests": selected,
        "requires_full_suite": requires_full_suite,
        "reasons": sorted(set(reasons)),
        "protected_risks": [
            "changed_contract_behavior",
            "success_claim_fail_closed",
            "runtime_security_fail_closed",
            "paid_resource_admission_fail_closed",
            "ci_release_policy",
        ],
    }


def write_plan(plan: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_github_output(plan: dict[str, object], path: Path) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            "requires_full_suite="
            + ("true" if plan["requires_full_suite"] else "false")
            + "\n"
        )
        stream.write(f"selected_test_count={len(plan['selected_tests'])}\n")


def run_pytest(
    root: Path,
    *,
    selected_tests: Sequence[str],
    timeout_seconds: int,
    junit_path: Path | None,
) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-m",
        "not slow and not gpu",
    ]
    if junit_path is not None:
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        command.append(f"--junitxml={junit_path}")
    command.extend(selected_tests)

    process = subprocess.Popen(command, cwd=root, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        print(
            f"[impacted-tests] wall-time budget exceeded: {timeout_seconds}s",
            file=sys.stderr,
        )
        return 124


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--no-worktree", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plan-output", type=Path, default=Path("output/impacted-test-plan.json"))
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    paths = args.changed_file or changed_files(
        ROOT,
        base=args.base,
        head=args.head,
        include_worktree=not args.no_worktree,
    )
    plan = build_plan(ROOT, paths)
    write_plan(plan, args.plan_output)
    if args.github_output:
        write_github_output(plan, args.github_output)

    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.plan_only:
        return 0
    return run_pytest(
        ROOT,
        selected_tests=plan["selected_tests"],
        timeout_seconds=args.timeout_seconds,
        junit_path=args.junit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
