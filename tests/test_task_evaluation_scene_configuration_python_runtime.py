from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_python_runtime as runtime_module

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


def _wheel(
    distribution: str,
    module: str,
    *,
    module_body: str = "PROVIDER_RUNTIME_FIXTURE = True\n",
) -> tuple[str, bytes]:
    filename = f"{distribution.replace('-', '_')}-1.0.0-py3-none-any.whl"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(f"{module}/__init__.py", module_body)
        archive.writestr(
            f"{distribution.replace('-', '_')}-1.0.0.dist-info/METADATA",
            f"Name: {distribution}\nVersion: 1.0.0\n",
        )
    return filename, stream.getvalue()


def _lock(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    agents_name, agents = _wheel("openai-agents", "provider_agents_fixture")
    pydantic_name, pydantic = _wheel("pydantic", "provider_pydantic_fixture")
    usd_name, usd = _wheel(
        "usd-core",
        "pxr",
        module_body="class Usd:\n    pass\n",
    )
    values = {agents_name: agents, pydantic_name: pydantic, usd_name: usd}

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
        + row("pydantic", pydantic_name, pydantic)
        + row("usd-core", usd_name, usd),
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
    assert "runtime_profile" not in manifest  # Legacy base manifest remains byte-compatible.
    assert "required_imports" not in manifest
    assert {row["name"] for row in manifest["requirements"]} == {
        "openai-agents",
        "pydantic",
        "usd-core",
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
        assert importlib.import_module("pxr").Usd
    finally:
        sys.path.remove(str(runtime))
        sys.modules.pop("provider_agents_fixture", None)
        sys.modules.pop("provider_pydantic_fixture", None)
        sys.modules.pop("pxr", None)


def test_real_lock_closes_agents_sdk_pydantic_and_usd_for_python_312() -> None:
    lockfile = Path(__file__).resolve().parents[1] / "uv.lock"
    plan = plan_scene_configuration_python_wheelhouse(lockfile.read_bytes())
    names = {row["name"] for row in plan["requirements"]}

    assert {
        "openai-agents",
        "openai",
        "pydantic",
        "pydantic-core",
        "usd-core",
    } <= names
    usd_wheel = next(
        row for row in plan["wheels"] if row["distribution"] == "usd-core"
    )
    assert "cp312" in usd_wheel["filename"]
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


def _astra_lock(tmp_path: Path, *, broken_module: str | None = None) -> tuple[Path, dict[str, bytes]]:
    """Tiny wheels exercise real isolated import validation without native binaries."""
    packages = {
        "openai-agents": {"agents/__init__.py": "class Agent: pass\n"},
        "usd-core": {"pxr/__init__.py": "", "pxr/Usd.py": "class Stage: pass\n"},
        "build123d": {"build123d/__init__.py": "from OCP import KERNEL\n"},
        "cadquery-ocp-novtk": {"OCP/__init__.py": "KERNEL = True\n"},
        "langgraph": {"langgraph/__init__.py": "", "langgraph/graph.py": "class StateGraph: pass\n"},
        "trimesh": {"trimesh/__init__.py": "class Trimesh: pass\n"},
        "pillow": {"PIL/__init__.py": "", "PIL/Image.py": "class Image: pass\n"},
    }
    dependencies = {"build123d": "cadquery-ocp-novtk"}
    rows, bodies = [], {}
    for name, files in packages.items():
        if broken_module == name:
            key = next(iter(files))
            files[key] = "raise ImportError('fixture missing required native dependency')\n"
        filename = f"{name.replace('-', '_')}-1.0.0-py3-none-any.whl"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as wheel:
            for filename_in_wheel, body in files.items():
                wheel.writestr(filename_in_wheel, body)
            wheel.writestr(f"{name.replace('-', '_')}-1.0.0.dist-info/METADATA", f"Name: {name}\nVersion: 1.0.0\n")
        body = stream.getvalue()
        bodies[filename] = body
        dependency = 'dependencies = [{name = "cadquery-ocp-novtk"}]\n' if name in dependencies else ""
        rows.append(f'[[package]]\nname = "{name}"\nversion = "1.0.0"\n{dependency}'
                    f'wheels = [{{url = "https://files.pythonhosted.org/packages/{filename}", '
                    f'hash = "sha256:{hashlib.sha256(body).hexdigest()}", size = {len(body)}}}]\n')
    lock = tmp_path / "astra.lock"
    lock.write_text("version = 1\n" + "".join(rows))
    return lock, bodies


@pytest.fixture(autouse=True)
def fixture_shipped_astra_source(tmp_path, monkeypatch):
    root = tmp_path / "shipped_source"
    package = root / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "fixture_stage.py").write_text("from PIL import Image\nimport build123d\n")
    monkeypatch.setattr(runtime_module, "_ASTRA_SOURCE_ROOT", root)
    monkeypatch.setattr(runtime_module, "_ASTRA_STAGE_MODULES", ("blueprint_pipeline.fixture_stage",))
    return root


def _build_astra(tmp_path, *, broken_module=None):
    lock, bodies = _astra_lock(tmp_path, broken_module=broken_module)
    root = tmp_path / "wheelhouse"
    manifest = build_scene_configuration_python_wheelhouse(lockfile_path=lock, output_root=root,
        downloader=lambda url, **kw: bodies[Path(url).name], profile="astra_asset_authoring")
    return root, manifest


def test_astra_profile_closes_locked_cad_and_langgraph_with_minimal_base_roots():
    lock = Path(__file__).parents[1] / "uv.lock"
    plan = plan_scene_configuration_python_wheelhouse(lock.read_bytes(), profile="astra_asset_authoring")
    versions = {row["name"]: row["version"] for row in plan["requirements"]}
    assert {key: versions[key] for key in ["openai-agents", "usd-core", "build123d", "langgraph",
            "langgraph-checkpoint", "langchain-core", "langgraph-sdk", "trimesh"]} == {
        "openai-agents": "0.19.1", "usd-core": "26.5", "build123d": "0.11.1", "langgraph": "0.2.76",
        "langgraph-checkpoint": "2.1.2", "langchain-core": "0.3.86", "langgraph-sdk": "0.1.74", "trimesh": "4.12.2",
    }
    ocp = next(row for row in plan["wheels"] if row["distribution"] == "cadquery-ocp-novtk")
    assert "cp312-cp312-manylinux_2_31_x86_64" in ocp["filename"]
    base = plan_scene_configuration_python_wheelhouse(lock.read_bytes())
    assert {row["name"] for row in base["requirements"]}.isdisjoint({"build123d", "langgraph", "pillow"})
    assert "pillow" in versions
    assert {"cryptography", "cffi", "pycparser"} <= versions.keys()  # Activated mcp -> pyjwt[crypto].
    with pytest.raises(ValueError, match="root_set_mismatch"):
        plan_scene_configuration_python_wheelhouse(lock.read_bytes(), root_distributions=["openai-agents"])
    with pytest.raises(ValueError, match="profile_invalid"):
        plan_scene_configuration_python_wheelhouse(lock.read_bytes(), profile="arbitrary-extra")


def test_astra_materialization_imports_sealed_profile_and_rejects_wrong_profile(tmp_path):
    root, manifest = _build_astra(tmp_path)
    assert manifest["runtime_profile"] == "astra_asset_authoring"
    assert manifest["root_distributions"] == ["openai-agents", "usd-core", "build123d", "langgraph", "trimesh", "pillow"]
    assert validate_scene_configuration_python_wheelhouse(root=root, profile="astra_asset_authoring") == manifest
    with pytest.raises(ValueError, match="manifest_invalid|profile_mismatch"):
        validate_scene_configuration_python_wheelhouse(root=root)
    with pytest.raises(ValueError, match="manifest_invalid|profile_mismatch"):
        materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path/"wrong",
            runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64")
    installed = materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path/"installed",
        runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64", profile="astra_asset_authoring")
    assert (installed / "build123d/__init__.py").is_file()
    assert not list(installed.rglob("*.pyc"))


@pytest.mark.parametrize("missing", ["build123d", "langgraph", "trimesh", "pillow"])
def test_missing_astra_root_wheel_is_refused_even_with_rehashed_inventory(tmp_path, missing):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root, manifest = _build_astra(tmp_path)
    removed = next(row for row in manifest["wheels"] if row["distribution"] == missing)
    (root/"wheels"/removed["filename"]).unlink()
    manifest["wheels"] = [row for row in manifest["wheels"] if row["distribution"] != missing]
    manifest["requirements"] = [row for row in manifest["requirements"] if row["name"] != missing]
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (root/MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="profile_inventory_incomplete"):
        materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path/"installed",
            runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64", profile="astra_asset_authoring")
    assert not (tmp_path/"installed").exists()


@pytest.mark.parametrize("broken", ["build123d", "langgraph", "cadquery-ocp-novtk", "trimesh", "pillow"])
def test_astra_import_failure_refuses_promotion_before_any_agent_call(tmp_path, broken):
    root, _ = _build_astra(tmp_path, broken_module=broken)
    with pytest.raises(ValueError, match="import_preflight_failed"):
        materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path/"installed",
            runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64", profile="astra_asset_authoring")
    assert not (tmp_path/"installed").exists()
    assert not (tmp_path/".installed.staging").exists()


def test_astra_wrong_wheel_abi_is_refused_before_import(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    root, manifest = _build_astra(tmp_path)
    row = manifest["wheels"][0]
    before = row["filename"]
    row["filename"] = before.replace("py3-none-any", "cp311-cp311-manylinux_2_31_x86_64")
    (root/"wheels"/before).rename(root/"wheels"/row["filename"])
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    (root/MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wheel_abi_or_identity_invalid"):
        validate_scene_configuration_python_wheelhouse(root=root, profile="astra_asset_authoring")


def test_missing_shipped_stage_dependency_is_detected_before_runtime_promotion(tmp_path, fixture_shipped_astra_source):
    (fixture_shipped_astra_source / "blueprint_pipeline/fixture_stage.py").write_text("import absent_stage_dependency\n")
    root, _ = _build_astra(tmp_path)
    with pytest.raises(ValueError, match="absent_stage_dependency"):
        materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path / "installed",
            runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64", profile="astra_asset_authoring")
    assert not (tmp_path / "installed").exists()


def test_stage_cannot_satisfy_dependencies_from_an_unsealed_global_root(tmp_path, fixture_shipped_astra_source):
    global_root = tmp_path / "global_packages"
    global_root.mkdir()
    (global_root / "unexpected_global_dependency.py").write_text("VALUE = True\n")
    (fixture_shipped_astra_source / "blueprint_pipeline/fixture_stage.py").write_text(
        f"import sys\nsys.path.append({str(global_root)!r})\nimport unexpected_global_dependency\n")
    root, _ = _build_astra(tmp_path)
    with pytest.raises(ValueError, match="global_python_module_not_admitted"):
        materialize_scene_configuration_python_runtime(wheelhouse_root=root, output_root=tmp_path / "installed",
            runtime_python=(3, 12), runtime_platform="linux", runtime_machine="x86_64", profile="astra_asset_authoring")
