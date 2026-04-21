"""Shared utilities for the autoresearch harness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def normalize_relpath(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def copy_relative_files(
    source_root: Path,
    destination_root: Path,
    relative_paths: Sequence[str],
    *,
    allow_missing: bool = False,
) -> None:
    for relative_path in relative_paths:
        normalized = normalize_relpath(relative_path)
        source_path = source_root / normalized
        destination_path = destination_root / normalized
        if not source_path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing required source file: {source_path}")
        ensure_dir(destination_path.parent)
        shutil.copy2(source_path, destination_path)


def line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def load_target_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    payload["_manifest_path"] = str(manifest_path.resolve())
    validate_target_manifest(payload)
    return payload


def skill_root_for_manifest(manifest: Mapping[str, Any]) -> str:
    mutable_paths = manifest.get("mutable_paths", [])
    if not mutable_paths:
        raise ValueError("Target manifest must include at least one mutable path.")
    return str(Path(str(mutable_paths[0])).parent.as_posix())


def validate_target_manifest(manifest: Mapping[str, Any]) -> None:
    target_skill = str(manifest.get("target_skill") or "").strip()
    if not target_skill:
        raise ValueError("Target manifest is missing target_skill.")

    mutable_paths = [normalize_relpath(item) for item in manifest.get("mutable_paths", [])]
    if not mutable_paths:
        raise ValueError("Target manifest must include mutable_paths.")

    skill_root = skill_root_for_manifest({"mutable_paths": mutable_paths})
    expected_suffix = f"/{target_skill}"
    if not skill_root.endswith(expected_suffix):
        raise ValueError(
            f"Mutable skill root '{skill_root}' does not match target skill '{target_skill}'."
        )

    all_mutable = mutable_paths + [
        normalize_relpath(item) for item in manifest.get("optional_mutable_paths", [])
    ]
    for relative_path in all_mutable:
        if not relative_path.startswith(f"{skill_root}/") and relative_path != skill_root:
            raise ValueError(
                f"Mutable path '{relative_path}' is outside the target skill directory '{skill_root}'."
            )

    for test_path in manifest.get("locked_harness_tests", []):
        normalized = normalize_relpath(test_path)
        if not normalized.startswith("autoresearch/tests/"):
            raise ValueError(f"Locked harness test must live under autoresearch/tests: {test_path}")

    for case in manifest.get("eval_cases", []):
        fixture_root = normalize_relpath(str(case.get("fixture_root") or ""))
        if not fixture_root.startswith(f"autoresearch/fixtures/{target_skill}/"):
            raise ValueError(
                f"Fixture root '{fixture_root}' must live under autoresearch/fixtures/{target_skill}/"
            )


@dataclass(frozen=True)
class PytestSummary:
    tests: int
    passed: int
    failed: int
    skipped: int
    exit_code: int
    paths: list[str]
    stdout: str
    stderr: str
    junit_xml: str

    @property
    def pass_rate(self) -> float:
        if self.tests <= 0:
            return 0.0
        return self.passed / float(self.tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests": self.tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "exit_code": self.exit_code,
            "selected_tests": list(self.paths),
            "pass_rate": round(self.pass_rate, 6),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "junit_xml": self.junit_xml,
        }


def _parse_junit_totals(xml_path: Path) -> tuple[int, int, int, int]:
    if not xml_path.is_file():
        return 0, 0, 0, 0
    root = ET.parse(xml_path).getroot()
    suites: list[ET.Element]
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
    if not suites:
        return 0, 0, 0, 0
    tests = failures = errors = skipped = 0
    for suite in suites:
        tests += int(float(suite.attrib.get("tests", 0)))
        failures += int(float(suite.attrib.get("failures", 0)))
        errors += int(float(suite.attrib.get("errors", 0)))
        skipped += int(float(suite.attrib.get("skipped", 0)))
    return tests, failures, errors, skipped


def _pytest_interpreters() -> list[str]:
    interpreters = [sys.executable]
    for candidate in ("python", "python3"):
        resolved = shutil.which(candidate)
        if resolved and resolved not in interpreters:
            interpreters.append(resolved)
    return interpreters


def _should_retry_pytest_interpreter(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}"
    return (
        "ModuleNotFoundError: No module named 'blueprint_contracts'" in combined
        or "No module named 'blueprint_contracts'" in combined
    )


def run_pytest(
    paths: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] | None = None,
) -> PytestSummary:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})

    interpreter_candidates = _pytest_interpreters()
    last_summary: PytestSummary | None = None

    for index, interpreter in enumerate(interpreter_candidates):
        with tempfile.TemporaryDirectory(prefix="autoresearch-pytest-") as tmp_dir:
            xml_path = Path(tmp_dir) / "pytest-report.xml"
            command = [
                interpreter,
                "-m",
                "pytest",
                "-q",
                "--junitxml",
                str(xml_path),
                *list(extra_args or []),
                *[str(item) for item in paths],
            ]
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=merged_env,
                text=True,
                capture_output=True,
                check=False,
            )
            tests, failures, errors, skipped = _parse_junit_totals(xml_path)
            passed = max(0, tests - failures - errors - skipped)
            summary = PytestSummary(
                tests=tests,
                passed=passed,
                failed=failures + errors,
                skipped=skipped,
                exit_code=int(completed.returncode),
                paths=[str(item) for item in paths],
                stdout=completed.stdout,
                stderr=completed.stderr,
                junit_xml=str(xml_path),
            )
            last_summary = summary
            if completed.returncode == 0:
                return summary
            if index + 1 < len(interpreter_candidates) and _should_retry_pytest_interpreter(
                completed.stdout,
                completed.stderr,
            ):
                continue
            return summary

    if last_summary is None:
        raise RuntimeError("run_pytest did not execute any interpreter candidates.")
    return last_summary
