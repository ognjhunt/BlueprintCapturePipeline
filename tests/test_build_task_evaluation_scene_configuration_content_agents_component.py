from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    validate_scene_configuration_component_package,
)
from scripts import (
    build_task_evaluation_scene_configuration_content_agents_component as subject,
)


ROOT = Path(__file__).resolve().parents[1]


def _commit(root: Path) -> tuple[str, str]:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, tree


def test_builds_released_content_agents_component_without_scene_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    copies = (
        "scripts/run_task_evaluation_scene_configuration_content_agents_component.sh",
        "scripts/run_adp_content_agents_provider_runtime.sh",
        "scripts/adp_content_agents_provider_runner.py",
        "src/blueprint_pipeline/provider_archive.py",
        "src/blueprint_pipeline/content_agents_model_compatibility.py",
        "docs/arm_decision_proof_v1/assets/adp009a_content_agents_material.vast.yaml",
        "docs/arm_decision_proof_v1/assets/adp009a_content_agents_texture.vast.yaml",
        "docs/arm_decision_proof_v1/assets/adp009a_content_agents_physics.vast.yaml",
    )
    for relative in copies:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if (ROOT / relative).stat().st_mode & 0o111:
            destination.chmod(0o755)
    blueprint_commit, _blueprint_tree = _commit(repository)

    upstream = tmp_path / "usd-content-agents"
    material = (
        upstream
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    )
    material.parent.mkdir(parents=True)
    material.write_text("materials: {}\n", encoding="utf-8")
    upstream_commit, upstream_tree = _commit(upstream)
    monkeypatch.setattr(subject, "SOURCE_COMMIT", upstream_commit)
    monkeypatch.setattr(subject, "SOURCE_TREE", upstream_tree)
    monkeypatch.setattr(subject, "SOURCE_VERSION", "test-version")
    git_commands: list[list[str]] = []
    original_run = subprocess.run

    def recorded_run(argv, *args, **kwargs):
        if argv and argv[0] == "git":
            git_commands.append(list(argv))
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(subject.subprocess, "run", recorded_run)

    output = tmp_path / "component"
    value = subject.build_content_agents_scene_configuration_component(
        repository_root=repository,
        expected_blueprint_commit=blueprint_commit,
        content_agents_root=upstream,
        output_root=output,
    )

    assert value == validate_scene_configuration_component_package(
        root=output,
        expected_adapter_id="content_agents_rigid_replacement",
    )
    assert (output / "content_agents_source.zip").is_file()
    assert (output / "content_agents_source_receipt.json").is_file()
    assert (output / "run").stat().st_mode & 0o111
    provider_runtime = output / "run_adp_content_agents_provider_runtime.sh"
    assert provider_runtime.stat().st_mode & 0o111
    provider_runtime_inventory = next(
        row
        for row in value["files"]
        if row["relative_path"] == provider_runtime.name
    )
    assert provider_runtime_inventory["executable"] is True
    assert not any("839873" in path.read_text(errors="ignore") for path in output.rglob("*.*"))
    archive_command = next(command for command in git_commands if "archive" in command)
    assert archive_command[:5] == [
        "git",
        "-c",
        f"safe.directory={upstream.resolve()}",
        "-C",
        str(upstream.resolve()),
    ]
