#!/usr/bin/env python3
"""Validate the Pipeline Python interpreter matrix against repo config."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "CI_PYTHON_INTERPRETER_MATRIX.json"
MATRIX_SCHEMA_VERSION = "blueprint_python_interpreter_matrix.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _collect_python_versions(value: Any) -> list[str]:
    versions: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "python-version":
                versions.append(str(item))
            else:
                versions.extend(_collect_python_versions(item))
    elif isinstance(value, list):
        for item in value:
            versions.extend(_collect_python_versions(item))
    return versions


def _workflow_python_versions(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _collect_python_versions(payload)


def _current_minor() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _normalize_specifier(value: str) -> str:
    return "".join(str(value or "").split())


def validate(*, assert_current: bool = False, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    matrix_path = root / "docs" / "CI_PYTHON_INTERPRETER_MATRIX.json"
    matrix = _read_json(matrix_path)
    canonical = str(matrix.get("canonical_launch_evidence_python") or "")
    requires_python = str(matrix.get("package_requires_python") or "")
    supported = [str(item) for item in matrix.get("package_supported_python") or []]

    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        errors.append("matrix_schema_version_mismatch")
    if not canonical:
        errors.append("canonical_launch_evidence_python_missing")
    if canonical not in supported:
        errors.append("canonical_python_not_listed_as_supported")

    pyproject = _read_toml(root / "pyproject.toml")
    project = dict(pyproject.get("project") or {})
    if _normalize_specifier(str(project.get("requires-python") or "")) != _normalize_specifier(requires_python):
        errors.append(
            f"pyproject_requires_python_mismatch:{project.get('requires-python')}!={requires_python}"
        )
    classifiers = {str(item) for item in project.get("classifiers") or []}
    for version in supported:
        expected = f"Programming Language :: Python :: {version}"
        if expected not in classifiers:
            errors.append(f"pyproject_classifier_missing:{expected}")
    if "Programming Language :: Python :: 3.13" in classifiers:
        errors.append("pyproject_classifier_includes_noncanonical_3.13")

    lock_head = (root / "uv.lock").read_text(encoding="utf-8").splitlines()[:8]
    lock_requires = next(
        (line.split("=", 1)[1].strip().strip('"') for line in lock_head if line.startswith("requires-python")),
        "",
    )
    if _normalize_specifier(lock_requires) != _normalize_specifier(requires_python):
        errors.append(f"uv_lock_requires_python_mismatch:{lock_requires}!={requires_python}")

    for workflow in matrix.get("ci_workflows") or []:
        workflow_path = root / str(workflow.get("path") or "")
        expected_version = str(workflow.get("python_version") or "")
        if not workflow_path.is_file():
            errors.append(f"workflow_missing:{workflow_path}")
            continue
        versions = _workflow_python_versions(workflow_path)
        if not versions:
            errors.append(f"workflow_python_version_missing:{workflow.get('path')}")
        unexpected = sorted({version for version in versions if version != expected_version})
        if unexpected:
            errors.append(
                f"workflow_python_version_mismatch:{workflow.get('path')}:{unexpected}!={expected_version}"
            )
        if expected_version != canonical:
            errors.append(
                f"workflow_not_on_canonical_python:{workflow.get('path')}:{expected_version}!={canonical}"
            )

    required_checks_doc = (root / "docs" / "CI_REQUIRED_CHECKS.md").read_text(encoding="utf-8")
    if "docs/CI_PYTHON_INTERPRETER_MATRIX.json" not in required_checks_doc:
        errors.append("ci_required_checks_missing_interpreter_matrix_reference")
    if f"Python `{canonical}`" not in required_checks_doc:
        errors.append("ci_required_checks_missing_canonical_python")

    if assert_current and _current_minor() != canonical:
        errors.append(f"current_python_not_canonical:{_current_minor()}!={canonical}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assert-current", action="store_true")
    args = parser.parse_args(argv)
    errors = validate(assert_current=args.assert_current)
    if errors:
        for error in errors:
            print(f"[python-matrix] ERROR {error}", file=sys.stderr)
        return 1
    print("[python-matrix] ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
