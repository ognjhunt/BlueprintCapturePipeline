from __future__ import annotations

import json
from pathlib import Path
import subprocess

from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    validate_scene_configuration_component_package,
)
from scripts.build_task_evaluation_scene_configuration_artifixer_component import (
    build_artifixer_scene_configuration_component,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path) -> tuple[str, str]:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def test_packages_released_artifixer_source_for_every_scene(tmp_path: Path, monkeypatch) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    repository = tmp_path / "blueprint"
    names = [
        "scripts/run_task_evaluation_scene_configuration_artifixer_component.sh",
        "scripts/run_public_scene_artifixer3d.sh",
        "scripts/public_scene_artifixer3d_runner.py",
        "src/blueprint_pipeline/__init__.py",
        "src/blueprint_pipeline/image_editor_backend_registry.py",
        "docs/arm_decision_proof_v1/manifests/image_editor_backends.v1.json",
    ]
    for name in names:
        source = source_repo / name
        destination = repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if source.stat().st_mode & 0o111:
            destination.chmod(0o755)
    blueprint_commit, _blueprint_tree = _commit(repository)
    artifixer = tmp_path / "artifixer"
    artifixer.mkdir()
    (artifixer / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (artifixer / "model_eval").mkdir()
    (artifixer / "model_eval/run_inference.py").write_text("# released fixture\n", encoding="utf-8")
    artifixer_commit, artifixer_tree = _commit(artifixer)
    import scripts.build_task_evaluation_scene_configuration_artifixer_component as subject

    monkeypatch.setattr(subject, "ARTIFIXER_COMMIT", artifixer_commit)
    monkeypatch.setattr(subject, "ARTIFIXER_TREE", artifixer_tree)
    output = tmp_path / "component"
    value = build_artifixer_scene_configuration_component(
        repository_root=repository,
        expected_blueprint_commit=blueprint_commit,
        artifixer_root=artifixer,
        output_root=output,
    )

    reopened = validate_scene_configuration_component_package(
        root=output,
        expected_adapter_id="artifixer3d_observed_object_removal",
    )
    source_receipt = json.loads(
        (output / "artifixer_source_receipt.json").read_text(encoding="utf-8")
    )
    assert value["package_digest"] == reopened["package_digest"]
    assert source_receipt["commit"] == artifixer_commit
    assert source_receipt["tree"] == artifixer_tree
    assert source_receipt["files"]
    assert (output / "run").stat().st_mode & 0o111
