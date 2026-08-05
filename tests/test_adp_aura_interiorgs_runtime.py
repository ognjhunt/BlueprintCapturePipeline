import importlib.util
import hashlib
import os
import sys
import zipfile
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


def test_aura_lama_checkpoint_is_rematerialized_immediately_before_execution(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    source = runtime / "AuraFusion360_official"
    runtime.mkdir()
    archive_path = runtime / "big-lama.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("big-lama/config.yaml", "training_model:\n  kind: default\n")
        archive.writestr("big-lama/models/best.ckpt", b"checkpoint-bytes")
    archive_sha256 = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()
    spec = {
        "lama": {
            "checkpoint_archive": "big-lama.zip",
            "checkpoint_sha256": archive_sha256,
        }
    }

    first = runner._materialize_lama_checkpoint(
        runtime=runtime, source=source, spec=spec
    )
    (source / "LaMa/big-lama/config.yaml").unlink()
    receipt_path = tmp_path / "evidence/materialization.json"
    second = runner._materialize_lama_checkpoint(
        runtime=runtime,
        source=source,
        spec=spec,
        receipt_path=receipt_path,
    )

    assert first["status"] == "accepted"
    assert second["status"] == "accepted"
    assert second["files"]["config"]["size_bytes"] > 0
    assert second["files"]["checkpoint"]["size_bytes"] > 0
    assert receipt_path.is_file()


def test_aura_lama_checkpoint_rejects_archive_without_config(tmp_path: Path) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive_path = runtime / "big-lama.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("big-lama/models/best.ckpt", b"checkpoint-bytes")
    archive_sha256 = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()

    receipt = runner._materialize_lama_checkpoint(
        runtime=runtime,
        source=runtime / "AuraFusion360_official",
        spec={
            "lama": {
                "checkpoint_archive": "big-lama.zip",
                "checkpoint_sha256": archive_sha256,
            }
        },
    )

    assert receipt["status"] == "blocked"
    assert "aurafusion360_interiorgs_lama_checkpoint_members_missing" in receipt["blockers"]


def test_aura_reference_lama_command_uses_absolute_verified_checkpoint_path(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "AuraFusion360_official"
    checkpoint_root = source / "LaMa/big-lama"
    (checkpoint_root / "models").mkdir(parents=True)
    (checkpoint_root / "config.yaml").write_text("training_model: {}\n")
    (checkpoint_root / "models/best.ckpt").write_bytes(b"checkpoint")

    command = runner._reference_lama_command(
        source=source,
        runtime=tmp_path / "runtime",
        lama_python=str(source / "LaMa/.venv/bin/python"),
    )

    assert command[2] == f"model.path={checkpoint_root.resolve()}"
    assert command[3] == f"indir={(tmp_path / 'runtime/reference_lama_input').resolve()}"
    assert command[4] == f"outdir={(tmp_path / 'runtime/reference_lama_output').resolve()}"


def test_aura_reference_lama_command_rejects_missing_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source = tmp_path / "AuraFusion360_official"
    checkpoint_root = source / "LaMa/big-lama"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "config.yaml").write_text("training_model: {}\n")

    try:
        runner._reference_lama_command(
            source=source,
            runtime=tmp_path / "runtime",
            lama_python=str(source / "LaMa/.venv/bin/python"),
        )
    except ValueError as exc:
        assert str(exc) == "aurafusion360_interiorgs_lama_checkpoint_path_unresolved"
    else:
        raise AssertionError("missing LaMa checkpoint bytes were not rejected")
