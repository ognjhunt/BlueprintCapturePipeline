#!/usr/bin/env python3
"""Select and run the smallest deterministic PR test set.

The selector is intentionally conservative. Directly related tests and a small
always-on sentinel set run for ordinary changes. A selection above the bounded
lane's budget runs every selected file in the wide lane, which has the full
lane's environment. Dependency-wide, unknown, or overly broad changes request the
separately governed full lane instead of silently pretending that a narrow
selection is sufficient.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_WORKER_COUNT = 4
MAX_CHANGED_FILES = 50
MAX_IMPACTED_TEST_FILES = 40
# Only an impact set this broad still needs the full suite. The 26 budget-only
# full suites of 2026-10-31 selected a median of 66 files (maximum 99), a median
# 8 of the suite's 161 test-minutes.
MAX_WIDE_TEST_FILES = 600
MAX_WIDE_SHARE_OF_TEST_FILES = 0.25
MAX_WIDE_SHARDS = 4
WIDE_SHARD_TARGET_SECONDS = 600
WIDE_LANE_WORKFLOW = ".github/workflows/impacted-wide-lane.yml"
DURATION_BASELINE = ".github/full-test-lane-duration-baseline.json"
FALLBACK_SECONDS_PER_TEST = 0.25
REVERSE_IMPORT_DEPTH = 2
# Source roots whose files are importable, as (path prefix, what to strip for the dotted name).
IMPORT_ROOTS = (("src/blueprint_pipeline/", "src/"), ("deploy/operator-door/operator_door/", "deploy/operator-door/"))
CONFIG_ROOT = "src/blueprint_pipeline/config/"
# Text, not a parse: 0.85 s against about 11 s for the 2,511 modules, and an import
# inside a function still counts. One inside a string only widens the selection.
IMPORT_STATEMENT = re.compile(
    r"^[ \t]*(?:from[ \t]+(?P<source>[.\w]+)[ \t]+import[ \t]+(?P<names>\([^)]*\)|[^\n#]*)"
    r"|import[ \t]+(?P<modules>[^\n#]+))",
    re.MULTILINE,
)
WORD = re.compile(r"\w+")

SENTINEL_TESTS = (
    "tests/test_success_claim_contracts.py::test_freshness_fails_closed_without_any_signal",
    "tests/test_runtime_security_controls.py::test_bounded_service_url_allows_only_loopback_plain_http",
    "tests/test_paid_resource_admission.py::test_shared_chokepoint_grants_only_exact_admitted_contract",
    "tests/test_paid_resource_allocator_verifier.py::test_unmanifested_script_mutator_is_rejected",
    # Whole-repository release checks that ci.yml otherwise runs only on push to main.
    "tests/test_paid_resource_allocator_verifier.py::test_repository_satisfies_paid_resource_allocator_verification",
    "tests/test_release_quality_governance.py::test_repository_source_governance_policy_is_satisfied",
    "tests/test_release_engineering_contracts.py::test_risk_based_verification_workflows_are_bounded",
    "tests/test_lane_writer_governance.py::test_reviewed_lane_writer_manifest_is_satisfied",
    "tests/test_materializer_reachability.py::test_unreachable_materializers_only_ever_decrease",
)

CROSS_CUTTING_FILES = {
    ".github/workflows/full-test-lane.yml",
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


def _dotted_module(path: str) -> str:
    """The name a file is imported by, or "" for a file nothing can import."""
    if not path.endswith(".py"):
        return ""
    for prefix, strip in IMPORT_ROOTS:
        if path.startswith(prefix):
            return path.removeprefix(strip).removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return ""


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

    # An unanchored basename also matches unrelated longer filenames (for
    # example service.py inside live_pipeline_intake_service.py). Preserve
    # literal file-loader references without creating that false dependency.
    tokens = {changed_path, f'"{path.name}"', f"'{path.name}'"}
    if path.name == "__init__.py":
        # Every package shares this basename, so only a test naming this exact
        # file (or, under src, importing this package) counts as coverage.
        tokens = {changed_path}
    module_patterns: list[str] = []
    module_name = ""
    module = _dotted_module(changed_path)
    if module:
        parent, _, module_name = module.rpartition(".")
        module_patterns = [
            rf"(?<![A-Za-z0-9_]){re.escape(module)}(?![A-Za-z0-9_])",
            rf"\bfrom\s+{re.escape(parent)}\s+import\s+(?:\(\s*)?{re.escape(module_name)}(?![A-Za-z0-9_])",
        ]
    elif changed_path.startswith("scripts/"):
        tokens.add(f"scripts/{path.name}")
        # A script test loads its subject by bare module name --
        # `_load("build_artifixer3d_live_profile")` -- which mentions neither
        # the path nor the filename, so the test was invisible here and every
        # such edit escalated to the full suite. Match the *quoted* stem: it is
        # exactly how the load is written, and it will not match a stem that
        # merely appears inside a longer identifier.
        tokens.update({f'"{stem}"', f"'{stem}'"})
    elif changed_path.startswith("tests/fixtures/"):
        tokens.add(path.parent.name)

    for test_path, source in test_sources.items():
        if test_path == "tests/test_impacted_test_selection.py":
            continue
        if (any(token and token in source for token in tokens)
                or (module_name in source
                    and any(re.search(pattern, source) for pattern in module_patterns))):
            candidates.add(test_path)
    return candidates


def _estimated_seconds(
    root: Path, *, files: Sequence[str], test_sources: dict[str, str]
) -> dict[str, float]:
    """Seconds per test file from the committed baseline; a planning hint only."""
    try:
        baseline = json.loads((root / DURATION_BASELINE).read_text(encoding="utf-8"))
        measured = {row["path"]: float(row["duration_seconds"]) for row in baseline["files"]}
        seconds_per_test = float(baseline["default_seconds_per_test"])
    except (OSError, ValueError, KeyError, TypeError):
        measured, seconds_per_test = {}, FALLBACK_SECONDS_PER_TEST
    return {
        path: measured.get(
            path, seconds_per_test * max(1, test_sources.get(path, "").count("def test_"))
        )
        for path in files
    }


def _wide_shards(
    root: Path, *, files: Sequence[str], test_sources: dict[str, str]
) -> list[dict[str, object]]:
    """Longest-first balanced shards, deterministic for one baseline and one selection."""
    if not files:
        return []
    estimates = _estimated_seconds(root, files=files, test_sources=test_sources)
    wanted = math.ceil(sum(estimates.values()) / WIDE_SHARD_TARGET_SECONDS)
    count = min(MAX_WIDE_SHARDS, max(1, wanted), len(files))
    loads = [0.0] * count
    assigned: list[list[str]] = [[] for _ in range(count)]
    for path, seconds in sorted(estimates.items(), key=lambda row: (-row[1], row[0])):
        index = loads.index(min(loads))
        assigned[index].append(path)
        loads[index] += seconds
    return [
        {"index": index, "estimated_seconds": round(loads[index], 3), "files": sorted(assigned[index])}
        for index in range(count)
    ]


def _imported_names(module: str, *, is_package: bool, source: str) -> set[str]:
    """Every dotted name this module's import statements could resolve to."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    names: set[str] = set()
    for statement in IMPORT_STATEMENT.finditer(source):
        if statement["modules"]:
            names.update(part.split(" as ")[0].strip() for part in statement["modules"].split(","))
            continue
        origin = statement["source"]
        level = len(origin) - len(origin.lstrip("."))
        if level:
            prefix = package[: len(package) - (level - 1)]
            origin = ".".join([*prefix, origin.lstrip(".")] if origin.lstrip(".") else prefix)
        names.add(origin)
        listed = re.sub(r"#[^\n]*", "", statement["names"]).strip("() \t\n")
        names.update(f"{origin}.{part.split(' as ')[0].strip()}" for part in listed.split(","))
    return names


def _importable_sources(root: Path) -> dict[str, str]:
    """Every importable file's source by repository path, and every script's."""
    sources: dict[str, str] = {}
    files = [*(root / "scripts").glob("*.py")]
    for prefix, _strip in IMPORT_ROOTS:
        files.extend((root / prefix).rglob("*.py"))
    for path in files:
        sources[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def _importers_by_file(sources: dict[str, str]) -> dict[str, set[str]]:
    by_module = {_dotted_module(path): path for path in sources if _dotted_module(path)}
    importers: dict[str, set[str]] = {}
    for path, source in sources.items():
        module = _dotted_module(path) or path.removesuffix(".py").replace("/", ".")
        for name in _imported_names(module, is_package=path.endswith("/__init__.py"), source=source):
            if name in by_module:
                importers.setdefault(by_module[name], set()).add(path)
    return importers


def _files_through_importers(root: Path, changed: Sequence[str]) -> dict[str, set[str]]:
    """For each file no test names: what imports it, within two steps, or what loads it."""
    sources = _importable_sources(root)
    importers = _importers_by_file(sources)
    through: dict[str, set[str]] = {}
    for changed_path in changed:
        if changed_path.startswith(CONFIG_ROOT) and not _dotted_module(changed_path):
            # A file or the directory directly under config/ holding it, named by its loader.
            name = changed_path.removeprefix(CONFIG_ROOT).split("/", 1)[0]
            through[changed_path] = {path for path, source in sources.items() if name in source}
            continue
        closure: set[str] = set()
        frontier = {changed_path}
        for _depth in range(REVERSE_IMPORT_DEPTH):
            frontier = {row for path in frontier for row in importers.get(path, ())} - closure - {changed_path}
            closure |= frontier
        through[changed_path] = closure
    return through


def _tests_of_files(paths: set[str], *, test_sources: dict[str, str]) -> dict[str, set[str]]:
    """_matching_tests for many files, reading each test's words only once."""
    by_word: dict[str, set[str]] = {}
    for path in paths:
        by_word.setdefault(_dotted_module(path).rpartition(".")[2] or Path(path).stem, set()).add(path)
    mentioning: dict[str, dict[str, str]] = {}
    for test_path, source in test_sources.items():
        for word in by_word.keys() & set(WORD.findall(source)):
            mentioning.setdefault(word, {})[test_path] = source
    matches: dict[str, set[str]] = {}
    for word, files in by_word.items():
        for path in files:
            direct = f"tests/test_{Path(path).stem}.py"
            candidates = mentioning.get(word, {})
            if direct in test_sources:
                candidates = {**candidates, direct: test_sources[direct]}
            matches[path] = _matching_tests(changed_path=path, test_sources=candidates)
    return matches


def build_plan(root: Path, paths: Sequence[str]) -> dict[str, object]:
    changed = _normalize_paths(paths)
    test_sources = _test_sources(root)
    impacted: set[str] = set()
    reasons: list[str] = []
    requires_full_suite = False
    needs_importers: list[str] = []

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

        if _dotted_module(changed_path) or changed_path.startswith(CONFIG_ROOT):
            needs_importers.append(changed_path)
        elif changed_path.startswith(("src/", "scripts/", "deploy/", "terraform/")):
            requires_full_suite = True
            reasons.append(f"unmapped_executable_surface:{changed_path}")
        else:
            reasons.append(f"non_executable_surface:{changed_path}")

    if needs_importers:
        through = _files_through_importers(root, needs_importers)
        tests_by_file = _tests_of_files(set().union(*through.values()), test_sources=test_sources)
        for changed_path in needs_importers:
            matches = set().union(*(tests_by_file[path] for path in through[changed_path]))
            if matches:
                impacted.update(matches)
                reasons.append(f"mapped_through_importers:{changed_path}:{len(matches)}")
            else:
                # A brand-new file nothing references, so nothing can say what it breaks.
                requires_full_suite = True
                reasons.append(f"unmapped_executable_surface:{changed_path}")

    requires_wide_impacted = False
    if len(impacted) > MAX_IMPACTED_TEST_FILES:
        ceiling = min(
            MAX_WIDE_TEST_FILES, int(MAX_WIDE_SHARE_OF_TEST_FILES * len(test_sources))
        )
        if requires_full_suite or len(impacted) > ceiling:
            requires_full_suite = True
            reasons.append(
                f"impacted_test_count_exceeds_budget:{len(impacted)}>{MAX_IMPACTED_TEST_FILES}"
            )
            if len(impacted) > ceiling:
                reasons.append(f"impacted_test_count_exceeds_wide_ceiling:{len(impacted)}>{ceiling}")
            impacted.clear()
        else:
            requires_wide_impacted = True
    if WIDE_LANE_WORKFLOW in changed and not requires_full_suite:
        # A change to the lane runs the lane, as a change to the full lane runs that.
        requires_wide_impacted = True
        reasons.append(f"wide_lane_changed:{WIDE_LANE_WORKFLOW}")

    wide_tests: list[str] = []
    if requires_wide_impacted:
        wide_tests = sorted(impacted)
        reasons.append(f"impacted_test_count_wide:{len(wide_tests)}")
        # The bounded lane stays bounded: it runs the sentinels, the wide lane the rest.
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
        "requires_wide_impacted": requires_wide_impacted,
        "wide_impacted_tests": wide_tests,
        "wide_shards": _wide_shards(root, files=wide_tests, test_sources=test_sources),
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
        stream.write(
            "requires_wide_impacted="
            + ("true" if plan["requires_wide_impacted"] else "false")
            + "\n"
        )
        matrix = [shard["index"] for shard in plan["wide_shards"]]
        stream.write(f"wide_shard_matrix={json.dumps(matrix, separators=(',', ':'))}\n")


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
        "-n",
        str(max(1, min(DEFAULT_WORKER_COUNT, os.cpu_count() or 1))),
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
