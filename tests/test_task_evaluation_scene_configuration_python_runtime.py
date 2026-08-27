from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_python_runtime import (
    TaskEvaluationSceneConfigurationPythonRuntimeError,
    materialize_scene_configuration_python_runtime,
)
from blueprint_pipeline.task_evaluation_scene_configuration_python_wheelhouse import (
    MANIFEST_NAME,
    build_scene_configuration_python_wheelhouse,
    plan_scene_configuration_python_wheelhouse,
    validate_scene_configuration_python_wheelhouse,
)


def _wheel(distribution: str, module: str) -> tuple[str, bytes]:
    filename = f"{distribution.replace('-', '_')}-1.0.0-py3-none-any.whl"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(f"{module}/__init__.py", "PROVIDER_RUNTIME_FIXTURE = True\n")
        archive.writestr(
            f"{distribution.replace('-', '_')}-1.0.0.dist-info/METADATA",
            f"Name: {distribution}\nVersion: 1.0.0\n",
        )
    return filename, stream.getvalue()


def _lock(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    agents_name, agents = _wheel("openai-agents", "provider_agents_fixture")
    pydantic_name, pydantic = _wheel("pydantic", "provider_pydantic_fixture")
    values = {agents_name: agents, pydantic_name: pydantic}

    def row(name: str, filename: str, body: bytes, *, dependency: str = "") -> str:
        dependencies = f'dependencies = [{{ name = "{dependency}" }}]\n' if dependency else ""
        digest = hashlib.sha256(body).hexdigest()
        return (
            "[[package]]\n"
            f'name = "{name}"\n'
            'version = "1.0.0"\n'
            f"{dependencies}"
            "wheels = [\n"
            f'  {{ url = "https://files.pythonhosted.org/packages/{filename}", '
            f'hash = "sha256:{digest}", size = {len(body)} }},\n'
            "]\n"
        )

    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        "version = 1\n"
        + row("openai-agents", agents_name, agents, dependency="pydantic")
        + row("pydantic", pydantic_name, pydantic),
        encoding="utf-8",
    )
    return lockfile, values


def test_builds_and_materializes_exact_provider_dependency_closure(
    tmp_path: Path,
) -> None:
    lockfile, bodies = _lock(tmp_path)
    output = tmp_path / "wheelhouse"

    manifest = build_scene_configuration_python_wheelhouse(
        lockfile_path=lockfile,
        output_root=output,
        downloader=lambda url, **_kwargs: bodies[Path(url).name],
    )
    reopened = validate_scene_configuration_python_wheelhouse(root=output)
    runtime = materialize_scene_configuration_python_runtime(
        wheelhouse_root=output,
        output_root=tmp_path / "provider-python",
        runtime_python=(3, 12),
        runtime_platform="linux",
        runtime_machine="x86_64",
    )

    assert manifest["manifest_digest"] == reopened["manifest_digest"]
    assert {row["name"] for row in manifest["requirements"]} == {
        "openai-agents",
        "pydantic",
    }
    sys.path.insert(0, str(runtime))
    try:
        assert (
            importlib.import_module("provider_agents_fixture").PROVIDER_RUNTIME_FIXTURE
            is True
        )
        assert (
            importlib.import_module("provider_pydantic_fixture").PROVIDER_RUNTIME_FIXTURE
            is True
        )
    finally:
        sys.path.remove(str(runtime))
        sys.modules.pop("provider_agents_fixture", None)
        sys.modules.pop("provider_pydantic_fixture", None)


def test_real_lock_closes_agents_sdk_and_pydantic_for_python_312() -> None:
    lockfile = Path(__file__).resolve().parents[1] / "uv.lock"
    plan = plan_scene_configuration_python_wheelhouse(lockfile.read_bytes())
    names = {row["name"] for row in plan["requirements"]}

    assert {"openai-agents", "openai", "pydantic", "pydantic-core"} <= names
    assert plan["wheels"]
    assert all("cp311" not in row["filename"] for row in plan["wheels"])


def test_materializer_refuses_tampered_or_wrong_platform_runtime(
    tmp_path: Path,
) -> None:
    lockfile, bodies = _lock(tmp_path)
    output = tmp_path / "wheelhouse"
    build_scene_configuration_python_wheelhouse(
        lockfile_path=lockfile,
        output_root=output,
        downloader=lambda url, **_kwargs: bodies[Path(url).name],
    )
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    wheel = output / "wheels" / manifest["wheels"][0]["filename"]
    wheel.write_bytes(wheel.read_bytes() + b"tamper")

    with pytest.raises(
        TaskEvaluationSceneConfigurationPythonRuntimeError,
        match="scene_configuration_python_wheel_invalid",
    ):
        materialize_scene_configuration_python_runtime(
            wheelhouse_root=output,
            output_root=tmp_path / "tampered-runtime",
            runtime_python=(3, 12),
            runtime_platform="linux",
            runtime_machine="x86_64",
        )
    with pytest.raises(
        TaskEvaluationSceneConfigurationPythonRuntimeError,
        match="scene_configuration_python_runtime_platform_mismatch",
    ):
        materialize_scene_configuration_python_runtime(
            wheelhouse_root=output,
            output_root=tmp_path / "wrong-platform-runtime",
            runtime_python=(3, 11),
            runtime_platform="linux",
            runtime_machine="x86_64",
        )
