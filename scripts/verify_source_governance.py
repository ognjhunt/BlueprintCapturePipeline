#!/usr/bin/env python3
"""Enforce module, CLI, and duplicated-claim growth budgets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


SCHEMA_VERSION = "blueprint.source_governance_policy.v1"
_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".conf",
    ".css",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".service",
    ".sh",
    ".tf",
    ".timer",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_GOVERNED_SOURCE_DIRS = (
    ".github",
    "autoresearch",
    "deploy",
    "docs",
    "scripts",
    "skillpacks",
    "src",
    "tests",
)
_GOVERNED_ROOT_FILES = (
    ".dockerignore",
    ".gcloudignore",
    ".gitignore",
    "AGENTS.md",
    "AUTONOMOUS_ORG.md",
    "CLAUDE.md",
    "Dockerfile",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "PLATFORM_CONTEXT.md",
    "README.md",
    "SECURITY.md",
    "VISION.md",
    "WORLD_MODEL_STRATEGY_CONTEXT.md",
    "docker-compose.yml",
    "main.py",
    "pyproject.toml",
    "requirements-geometry.txt",
    "requirements.txt",
    "ops/city-launch-runs/README.md",
    "uv.lock",
)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _governed_text_paths(root: Path) -> list[Path]:
    candidates = [root / name for name in _GOVERNED_ROOT_FILES if (root / name).is_file()]
    for directory_name in _GOVERNED_SOURCE_DIRS:
        directory = root / directory_name
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(
        {
            path
            for path in candidates
            if path.suffix.lower() in _TEXT_SUFFIXES and not path.is_symlink()
        }
    )


def validate_source_governance(
    *,
    root: Path,
    policy: Mapping[str, Any],
    today: date | None = None,
) -> dict[str, Any]:
    current_date = today or date.today()
    blockers: list[str] = []
    if policy.get("schema_version") != SCHEMA_VERSION:
        blockers.append("policy_schema_version_invalid")
    if len(str(policy.get("policy_owner") or "").strip()) < 3:
        blockers.append("policy_owner_missing")
    baseline_date: date | None = None
    try:
        parsed_baseline_date = date.fromisoformat(str(policy.get("baseline_date") or ""))
    except ValueError:
        blockers.append("baseline_date_invalid")
    else:
        baseline_date = parsed_baseline_date
        if parsed_baseline_date > current_date:
            blockers.append("baseline_date_future")
    try:
        review_by = date.fromisoformat(str(policy.get("review_by") or ""))
    except ValueError:
        blockers.append("review_by_invalid")
    else:
        if review_by < current_date:
            blockers.append("source_governance_policy_expired")
        if baseline_date is not None and review_by < baseline_date:
            blockers.append("review_by_precedes_baseline")
        if baseline_date is not None and review_by > baseline_date + timedelta(days=90):
            blockers.append("source_governance_review_window_exceeds_90_days")
    default_limit = policy.get("default_max_python_module_lines")
    if not isinstance(default_limit, int) or default_limit <= 0:
        blockers.append("default_module_limit_invalid")
        default_limit = 0
    grandfathered = _mapping(policy.get("grandfathered_module_line_limits"))
    source_root = root / "src" / "blueprint_pipeline"
    measured: dict[str, int] = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        count = _line_count(path)
        measured[relative] = count
        raw_limit = grandfathered.get(relative, default_limit)
        if not isinstance(raw_limit, int) or raw_limit <= 0:
            blockers.append(f"module_limit_invalid:{relative}")
        elif count > raw_limit:
            blockers.append(f"module_line_budget_exceeded:{relative}:{count}>{raw_limit}")
    for relative in sorted(set(grandfathered) - set(measured)):
        blockers.append(f"grandfathered_module_missing:{relative}")

    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
    except (OSError, UnicodeError, KeyError, tomllib.TOMLDecodeError):
        blockers.append("project_scripts_unreadable")
        scripts = {}
    cli_limit = policy.get("max_project_scripts")
    if not isinstance(cli_limit, int) or cli_limit < 0:
        blockers.append("project_script_limit_invalid")
    elif len(scripts) > cli_limit:
        blockers.append(f"project_script_budget_exceeded:{len(scripts)}>{cli_limit}")

    token_limits = _mapping(policy.get("claim_literal_maximums"))
    token_counts: dict[str, int] = {}
    source_texts = [path.read_text(encoding="utf-8") for path in sorted(source_root.rglob("*.py"))]
    for token, raw_limit in sorted(token_limits.items()):
        count = sum(text.count(token) for text in source_texts)
        token_counts[token] = count
        if not isinstance(raw_limit, int) or raw_limit < 0:
            blockers.append(f"claim_literal_limit_invalid:{token}")
        elif count > raw_limit:
            blockers.append(f"duplicated_claim_literal_budget_exceeded:{token}:{count}>{raw_limit}")

    characterization = _mapping(policy.get("characterization_tests"))
    for module in sorted(set(grandfathered) - set(characterization)):
        blockers.append(f"grandfathered_module_characterization_missing:{module}")
    for module, raw_tests in sorted(characterization.items()):
        tests = raw_tests if isinstance(raw_tests, list) else []
        if module not in measured:
            blockers.append(f"characterized_module_missing:{module}")
        if not tests:
            blockers.append(f"characterization_tests_missing:{module}")
        for test_path in tests:
            if not isinstance(test_path, str) or not (root / test_path).is_file():
                blockers.append(f"characterization_test_missing:{module}:{test_path}")

    raw_path_patterns = policy.get("forbidden_personal_path_patterns")
    path_patterns: list[re.Pattern[str]] = []
    if not isinstance(raw_path_patterns, list) or not raw_path_patterns:
        blockers.append("forbidden_personal_path_patterns_missing")
    else:
        for index, raw_pattern in enumerate(raw_path_patterns):
            if not isinstance(raw_pattern, str) or not raw_pattern:
                blockers.append(f"forbidden_personal_path_pattern_invalid:{index}")
                continue
            try:
                path_patterns.append(re.compile(raw_pattern))
            except re.error:
                blockers.append(f"forbidden_personal_path_pattern_invalid:{index}")
    governed_text_paths = _governed_text_paths(root)
    for path in governed_text_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            blockers.append(f"governed_text_unreadable:{path.relative_to(root).as_posix()}")
            continue
        for pattern in path_patterns:
            match = pattern.search(text)
            if match is None:
                continue
            relative = path.relative_to(root).as_posix()
            line_number = text.count("\n", 0, match.start()) + 1
            blockers.append(f"personal_absolute_path_forbidden:{relative}:{line_number}")
            break

    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.source_governance_gate.v1",
        "status": "passed" if not blockers else "blocked",
        "measured_python_module_count": len(measured),
        "grandfathered_module_count": len(grandfathered),
        "project_script_count": len(scripts),
        "personal_path_scanned_file_count": len(governed_text_paths),
        "claim_literal_counts": token_counts,
        "blockers": blockers,
        "claim_boundary": {
            "budgets_prevent_growth_but_do_not_complete_refactoring": True,
            "characterization_is_required_before_hotspot_splits": True,
            "new_duplicate_claim_decisions_are_blocked": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("docs/source_governance_policy.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[source-governance] ERROR unreadable_policy:{exc}", file=sys.stderr)
        return 1
    result = validate_source_governance(
        root=args.root.resolve(),
        policy=_mapping(policy),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        "[source-governance] "
        f"status={result['status']} modules={result['measured_python_module_count']} "
        f"clis={result['project_script_count']}"
    )
    for blocker in result["blockers"]:
        print(f"[source-governance] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
