import importlib.util
import os
import sys
from pathlib import Path


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "test_adp_aura_interiorgs_provider_runner",
            scripts / "adp_aura_interiorgs_provider_runner.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_aura_lama_runtime_isolates_legacy_opencv_distribution() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[1] / "scripts/run_adp_aura_interiorgs_provider_runtime.sh"
    ).read_text(encoding="utf-8")

    assert 'pip uninstall --python "${LAMA_PY}" opencv-python' in entrypoint
    assert "--reinstall --no-deps opencv-python-headless==4.5.5.64" in entrypoint
    assert 'metadata.version("opencv-python-headless") == "4.5.5.64"' in entrypoint
    assert 'numpy.__version__ == "1.21.6"' in entrypoint
    assert "aurafusion360_interiorgs_lama_opencv_validation_failed" in entrypoint


def test_aura_adapter_overlay_preserves_publisher_tree(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "adapter"
    destination = tmp_path / "publisher"
    (source / "Other-360/new-scene").mkdir(parents=True)
    (source / "Other-360/new-scene/train.config").write_text("new\n")
    (destination / "Other-360/publisher-scene").mkdir(parents=True)
    publisher = destination / "Other-360/publisher-scene/train.config"
    publisher.write_text("publisher\n")

    runner._copy_tree_additions(source, destination)

    assert publisher.read_text() == "publisher\n"
    assert (destination / "Other-360/new-scene/train.config").read_text() == "new\n"


def test_aura_adapter_overlay_rejects_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "adapter"
    destination = tmp_path / "publisher"
    source.mkdir()
    destination.mkdir()
    (source / "train.config").write_text("adapter\n")
    (destination / "train.config").write_text("publisher\n")

    try:
        runner._copy_tree_additions(source, destination)
    except ValueError as exc:
        assert str(exc) == "aurafusion360_interiorgs_adapter_path_conflict"
    else:
        raise AssertionError("adapter overwrite was not rejected")


def test_aura_subprocess_receives_explicit_environment(tmp_path: Path) -> None:
    runner = _load_runner()
    log_path = tmp_path / "environment.log"

    result = runner.shared._run(
        ["/usr/bin/env"],
        cwd=tmp_path,
        log_path=log_path,
        env={"AURA_LAMA_SOURCE_BOUND": "yes"},
    )

    assert result["returncode"] == 0
    assert "AURA_LAMA_SOURCE_BOUND=yes" in log_path.read_text()


def test_aura_lama_environment_binds_source_before_runtime_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    dependencies = tmp_path / "dependencies"
    lama_source = tmp_path / "LaMa"
    monkeypatch.setenv("PYTHONPATH", "inherited")

    aura_env, lama_env = runner._runtime_environments(
        dependencies=dependencies,
        lama_source=lama_source,
    )

    assert aura_env["PYTHONPATH"] == f"{dependencies}{os.pathsep}inherited"
    assert lama_env["PYTHONPATH"] == (
        f"{lama_source}{os.pathsep}{dependencies}{os.pathsep}inherited"
    )
    runner_source = (
        Path(__file__).resolve().parents[1] / "scripts/adp_aura_interiorgs_provider_runner.py"
    ).read_text(encoding="utf-8")
    assert "env=env" in runner_source
