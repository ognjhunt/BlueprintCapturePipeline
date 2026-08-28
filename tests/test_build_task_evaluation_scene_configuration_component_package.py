from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    validate_scene_configuration_component_package,
)
from scripts.build_task_evaluation_scene_configuration_component_package import (
    build_scene_configuration_component_package,
)


def test_builds_scene_neutral_exhaustive_component_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    driver = source / "run"
    driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    driver.chmod(0o755)
    (source / "public_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    immutable_asset = source / "immutable_asset.bin"
    immutable_asset.write_bytes(b"immutable model bytes")
    immutable_asset.chmod(0o444)
    output = tmp_path / "package"

    value = build_scene_configuration_component_package(
        adapter_id="content_agents_rigid_replacement",
        source_root=source,
        driver_entrypoint="run",
        source_repository="https://github.com/NVIDIA-Omniverse/usd-content-agents",
        source_commit="c" * 40,
        source_license="Apache-2.0",
        output_root=output,
    )

    assert value == validate_scene_configuration_component_package(
        root=output,
        expected_adapter_id="content_agents_rigid_replacement",
    )
    assert value["source_identity"]["scene_specific_source"] is False
    assert {row["relative_path"] for row in value["files"]} == {
        "immutable_asset.bin",
        "public_source.py",
        "run",
    }
    assert (output / "immutable_asset.bin").stat().st_ino == immutable_asset.stat().st_ino
    assert all(not path.stat().st_mode & 0o222 for path in output.rglob("*"))


def test_component_package_build_rejects_symlink_and_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    driver = source / "run"
    driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    driver.chmod(0o755)
    (source / "escape").symlink_to(driver)
    with pytest.raises(ValueError, match="source_symlink_forbidden"):
        build_scene_configuration_component_package(
            adapter_id="artifixer3d_observed_object_removal",
            source_root=source,
            driver_entrypoint="run",
            source_repository="https://github.com/nv-tlabs/ArtiFixer.git",
            source_commit="a" * 40,
            source_license="NVIDIA Source Code License",
            output_root=tmp_path / "package",
        )

    (source / "escape").unlink()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="output_exists"):
        build_scene_configuration_component_package(
            adapter_id="artifixer3d_observed_object_removal",
            source_root=source,
            driver_entrypoint="run",
            source_repository="https://github.com/nv-tlabs/ArtiFixer.git",
            source_commit="a" * 40,
            source_license="NVIDIA Source Code License",
            output_root=output,
        )
