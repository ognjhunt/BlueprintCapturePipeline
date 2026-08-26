from __future__ import annotations

import subprocess
from pathlib import Path

from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    validate_scene_configuration_component_package,
)
from scripts.build_task_evaluation_scene_configuration_native_import_component import (
    build_native_import_scene_configuration_component,
)


def test_native_import_component_is_release_bound_and_scene_neutral(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    driver = (
        repository
        / "src/blueprint_pipeline/task_evaluation_scene_configuration_native_import_driver.py"
    )
    driver.parent.mkdir(parents=True)
    driver.write_text("# exact native driver\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Blueprint Tests",
            "-c",
            "user.email=tests@blueprint.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    output = tmp_path / "component"

    built = build_native_import_scene_configuration_component(
        repository_root=repository,
        expected_blueprint_commit=commit,
        output_root=output,
    )

    reopened = validate_scene_configuration_component_package(
        root=output,
        expected_adapter_id="simready_native_import_qualification",
    )
    assert built == reopened
    assert built["source_identity"]["scene_specific_source"] is False
    assert built["network_policy"] == "disabled"
    assert (output / "run").stat().st_mode & 0o111

