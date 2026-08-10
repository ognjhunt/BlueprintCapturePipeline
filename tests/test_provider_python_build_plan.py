from __future__ import annotations

import json

import pytest

from blueprint_pipeline.provider_python_build_plan import (
    materialize_python_build_plan,
    validate_python_build_plan,
)


def _project(
    root, name: str, requirements: list[str], *, editable_source: bool = False
) -> str:
    path = root / name / "pyproject.toml"
    path.parent.mkdir(parents=True)
    source = (
        "[build-system]\n"
        + "requires = "
        + json.dumps(requirements)
        + "\n"
        + 'build-backend = "hatchling.build"\n'
    )
    if editable_source:
        source += (
            "\n[tool.uv.sources]\n"
            'world-understanding = { path = "../..", editable = true }\n'
        )
    path.write_text(source, encoding="utf-8")
    return path.relative_to(root).as_posix()


def test_build_plan_preflights_all_local_project_build_requirements(tmp_path) -> None:
    source = tmp_path / "source"
    root_project = _project(
        source, "root", ["hatchling<1.27.0", "uv-dynamic-versioning"]
    )
    service_project = _project(
        source,
        "service",
        ["setuptools>=68.0", "setuptools-scm>=8.0"],
        editable_source=True,
    )

    plan = materialize_python_build_plan(
        source_root=source,
        project_relative_paths=[root_project, service_project],
        destination=tmp_path / "plan.json",
    )

    assert plan["requirements"] == [
        "editables>=0.3",
        "hatchling<1.27.0",
        "setuptools-scm>=8.0",
        "setuptools>=68.0",
        "uv-dynamic-versioning",
    ]
    assert plan["dynamic_backend_requirements"] == ["editables>=0.3"]
    assert plan["build_isolation_required"] is False
    assert validate_python_build_plan(
        plan_path=tmp_path / "plan.json", source_root=source
    ) == plan


def test_build_plan_rejects_pyproject_drift_before_install(tmp_path) -> None:
    source = tmp_path / "source"
    project = _project(source, "root", ["hatchling<1.27.0"])
    materialize_python_build_plan(
        source_root=source,
        project_relative_paths=[project],
        destination=tmp_path / "plan.json",
    )
    (source / project).write_text(
        '[build-system]\nrequires = ["setuptools"]\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="provider_python_build_plan_source_drift"):
        validate_python_build_plan(
            plan_path=tmp_path / "plan.json", source_root=source
        )
