"""Digest-bound build-system dependency planning for provider Python projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "provider_python_build_dependency_plan.v1"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _dynamic_backend_requirements(document: Mapping[str, Any]) -> set[str]:
    uv_sources = ((document.get("tool") or {}).get("uv") or {}).get("sources", {})
    if isinstance(uv_sources, Mapping) and any(
        isinstance(spec, Mapping) and spec.get("editable") is True
        for spec in uv_sources.values()
    ):
        return {"editables>=0.3"}
    return set()


def materialize_python_build_plan(
    *,
    source_root: str | Path,
    project_relative_paths: Iterable[str],
    destination: str | Path,
) -> dict[str, Any]:
    """Freeze every local project's declared PEP 517 build requirement."""

    root = Path(source_root).expanduser().resolve()
    projects: list[dict[str, Any]] = []
    requirements: set[str] = set()
    dynamic_backend_requirements: set[str] = set()
    for relative in sorted(set(project_relative_paths)):
        path = (root / relative).resolve()
        if root not in path.parents or path.name != "pyproject.toml" or not path.is_file():
            raise ValueError("provider_python_build_project_invalid")
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        build = document.get("build-system") or {}
        declared = build.get("requires") or []
        if (
            not isinstance(build, Mapping)
            or not isinstance(declared, list)
            or not declared
            or any(not isinstance(item, str) or not item.strip() for item in declared)
        ):
            raise ValueError("provider_python_build_requirements_invalid")
        normalized = sorted(set(item.strip() for item in declared))
        requirements.update(normalized)
        inferred = _dynamic_backend_requirements(document)
        if inferred:
            # uv honors first-party editable path declarations even when the
            # top-level provider install is non-editable. Hatchling's editable
            # hook imports this package dynamically without declaring it in
            # build-system.requires (observed in NVIDIA Content Agents v0.5.2).
            dynamic_backend_requirements.update(inferred)
        projects.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "build_backend": str(build.get("build-backend") or ""),
                "requires": normalized,
            }
        )
    requirements.update(dynamic_backend_requirements)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "projects": projects,
        "requirements": sorted(requirements),
        "dynamic_backend_requirements": sorted(dynamic_backend_requirements),
        "build_isolation_required": False,
        "reason": "all_digest_bound_build_requirements_preinstalled_before_editable_install",
    }
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def validate_python_build_plan(
    *, plan_path: str | Path, source_root: str | Path
) -> dict[str, Any]:
    """Re-read exact pyprojects and reject drift before dependency install."""

    root = Path(source_root).expanduser().resolve()
    value = json.loads(Path(plan_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("provider_python_build_plan_invalid")
    projects = value.get("projects")
    requirements = value.get("requirements")
    if (
        not isinstance(projects, list)
        or not projects
        or not isinstance(requirements, list)
        or not requirements
    ):
        raise ValueError("provider_python_build_plan_invalid")
    observed: set[str] = set()
    observed_dynamic: set[str] = set()
    for row in projects:
        if not isinstance(row, Mapping):
            raise ValueError("provider_python_build_plan_invalid")
        path = (root / str(row.get("relative_path") or "")).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != row.get("sha256"):
            raise ValueError("provider_python_build_plan_source_drift")
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        declared = document.get("build-system", {}).get("requires", [])
        observed.update(str(item).strip() for item in declared)
        observed_dynamic.update(_dynamic_backend_requirements(document))
    observed.update(observed_dynamic)
    if sorted(observed_dynamic) != value.get("dynamic_backend_requirements"):
        raise ValueError("provider_python_build_plan_requirement_drift")
    if sorted(observed) != requirements:
        raise ValueError("provider_python_build_plan_requirement_drift")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_path")
    parser.add_argument("source_root")
    parser.add_argument("--requirements-out", required=True)
    args = parser.parse_args()
    try:
        plan = validate_python_build_plan(
            plan_path=args.plan_path, source_root=args.source_root
        )
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError):
        return 2
    output = Path(args.requirements_out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(str(item) for item in plan["requirements"]) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "materialize_python_build_plan",
    "validate_python_build_plan",
]
