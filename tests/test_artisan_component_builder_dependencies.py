"""ADP-009: the ARTISAN component builder's install checks (2026-10-13 production install)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_task_evaluation_scene_configuration_artisan_component.py"


def _builder():
    spec = importlib.util.spec_from_file_location("artisan_component_builder_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_index_local_label_satisfies_its_public_pin():
    """The pinned CUDA index installs torch 2.7.1+cu128 for the root torch==2.7.1."""
    builder = _builder()
    installed = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "absl-py": "2.4.0"}
    assert builder.missing_root_dependencies(
        ["torch==2.7.1", "torchvision==0.22.1", "absl-py==2.4.0"], installed) == []


def test_a_different_absent_or_unpinned_root_is_still_missing():
    builder = _builder()
    installed = {"numpy": "1.26.4", "torch": "2.7.2+cu128"}
    assert builder.missing_root_dependencies(
        ["numpy==2.0.0", "scipy==1.14.0", "torch==2.7.1", "fastapi>=0.1"], installed) == [
        "numpy==2.0.0", "scipy==1.14.0", "torch==2.7.1", "fastapi>=0.1"]


def test_root_names_compare_normalized():
    builder = _builder()
    assert builder.missing_root_dependencies(["Pillow==11.0.0", "typing_extensions==4.12.2"],
                                             {"pillow": "11.0.0", "typing-extensions": "4.12.2"}) == []


def test_the_isolated_import_preflight_tolerates_a_cold_runtime(tmp_path, monkeypatch):
    """A freshly copied runtime imported without bytecode took over 60 s under load (2026-10-13)."""
    import json
    import subprocess

    builder = _builder()
    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(
            {"status": "isolated_upstream_imports_passed"}), stderr="")
    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    builder._runtime_preflight(tmp_path)
    assert seen["timeout"] == builder.PREFLIGHT_TIMEOUT_SECONDS >= 300


def test_sealed_runtime_package_shares_bytes_and_survives_staging_cleanup(tmp_path):
    """Packaging must not double the large temporary runtime's disk footprint."""
    import shutil
    from scripts.build_task_evaluation_scene_configuration_component_package import build_scene_configuration_component_package
    from blueprint_pipeline.task_evaluation_scene_configuration_component_package import validate_scene_configuration_component_package

    staging = tmp_path / "owned-staging"
    staging.mkdir()
    driver = staging / "run"
    driver.write_text("#!/bin/sh\nexit 0\n")
    driver.chmod(0o755)
    payload = staging / "runtime.bin"
    original = b"locked vendor runtime\n" * 10000
    payload.write_bytes(original)
    payload.chmod(0o644)
    builder = _builder()
    builder._seal_staging_files(staging)
    output = tmp_path / "published-package"
    receipt = build_scene_configuration_component_package(adapter_id="content_agents_rigid_replacement",
        source_root=staging, driver_entrypoint="run", source_repository="https://github.com/ognjhunt/BlueprintCapturePipeline",
        source_commit="a" * 40, source_license="test", output_root=output)
    for source in (driver, payload):
        target = output / source.name
        assert source.stat().st_ino == target.stat().st_ino
        assert not target.stat().st_mode & 0o222
    assert (output / "run").stat().st_mode & 0o111
    shutil.rmtree(staging)
    assert (output / "runtime.bin").read_bytes() == original
    assert validate_scene_configuration_component_package(root=output,
        expected_adapter_id="content_agents_rigid_replacement") == receipt


def test_staging_seal_refuses_links_before_changing_any_permissions(tmp_path):
    import pytest
    staging = tmp_path / "owned-staging"
    staging.mkdir()
    vendor = tmp_path / "vendor.bin"
    vendor.write_bytes(b"vendor installation")
    vendor.chmod(0o644)
    (staging / "escape").symlink_to(vendor)
    with pytest.raises(ValueError, match="artisan_staging_symlink_forbidden"):
        _builder()._seal_staging_files(staging)
    assert vendor.stat().st_mode & 0o222
