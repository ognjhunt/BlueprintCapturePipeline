from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_scene_configuration_component_package as package_module
from blueprint_pipeline.task_evaluation_scene_configuration_component_package import (
    SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationComponentPackageError,
    validate_scene_configuration_component_package,
)
from tests.test_build_task_evaluation_scene_configuration_toolchain import (
    _component_packages,
)


def test_component_package_is_platform_bound_and_scene_neutral(tmp_path: Path) -> None:
    root = _component_packages(tmp_path)[
        "artifixer3d_observed_object_removal"
    ]

    value = validate_scene_configuration_component_package(
        root=root,
        expected_adapter_id="artifixer3d_observed_object_removal",
    )

    assert value["source_identity"]["scene_specific_source"] is False
    assert value["driver_entrypoint"] == "run"
    assert value["network_policy"] == "provider_and_openai_api"


def test_component_package_rejects_scene_specific_or_changed_bytes(
    tmp_path: Path,
) -> None:
    root = _component_packages(tmp_path)[
        "content_agents_rigid_replacement"
    ]
    manifest_path = root / f"{SCHEMA_VERSION}.json"
    manifest_path.chmod(0o644)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["source_identity"]["scene_specific_source"] = True
    value["package_digest"] = canonical_digest(
        value, digest_field="package_digest"
    )
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path.chmod(0o444)
    with pytest.raises(
        TaskEvaluationSceneConfigurationComponentPackageError,
        match="component_package_manifest_invalid",
    ):
        validate_scene_configuration_component_package(
            root=root,
            expected_adapter_id="content_agents_rigid_replacement",
        )

    value["source_identity"]["scene_specific_source"] = False
    value["package_digest"] = canonical_digest(
        value, digest_field="package_digest"
    )
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path.chmod(0o444)
    driver = root / "run"
    driver.chmod(0o755)
    driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    driver.chmod(0o555)
    with pytest.raises(
        TaskEvaluationSceneConfigurationComponentPackageError,
        match="component_package_inventory_invalid",
    ):
        validate_scene_configuration_component_package(
            root=root,
            expected_adapter_id="content_agents_rigid_replacement",
        )


def test_component_package_rejects_adapter_network_policy_mismatch(
    tmp_path: Path,
) -> None:
    root = _component_packages(tmp_path)[
        "content_agents_rigid_replacement"
    ]
    manifest_path = root / f"{SCHEMA_VERSION}.json"
    manifest_path.chmod(0o644)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["network_policy"] = "disabled"
    value["package_digest"] = canonical_digest(
        value, digest_field="package_digest"
    )
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    manifest_path.chmod(0o444)

    with pytest.raises(
        TaskEvaluationSceneConfigurationComponentPackageError,
        match="component_package_manifest_invalid",
    ):
        validate_scene_configuration_component_package(
            root=root,
            expected_adapter_id="content_agents_rigid_replacement",
        )


def _lookup(call):
    try:
        return call()
    except TaskEvaluationSceneConfigurationComponentPackageError as refusal:
        return str(refusal)


@pytest.mark.parametrize("value", [
    "plain.txt", "nested/deep/plain.txt", "./nested/deep/plain.txt", "nested//deep/plain.txt",
    "leaf-link-inside", "leaf-link-outside", "leaf-link-dangling", "dir-link-inside/plain.txt",
    "dir-link-outside/secret.txt", "missing.txt", "nested", "../secret.txt", "nested/../plain.txt", "/etc/hosts", "",
])
def test_the_inventory_lookup_agrees_with_resolving_the_whole_path(tmp_path: Path, value: str) -> None:
    """It asks about each directory once, and must not change a single answer."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    root = tmp_path / "package"
    (root / "nested" / "deep").mkdir(parents=True)
    (root / "plain.txt").write_text("plain", encoding="utf-8")
    (root / "nested" / "deep" / "plain.txt").write_text("deep", encoding="utf-8")
    (root / "leaf-link-inside").symlink_to(root / "plain.txt")
    (root / "leaf-link-outside").symlink_to(outside / "secret.txt")
    (root / "leaf-link-dangling").symlink_to(root / "absent")
    (root / "dir-link-inside").symlink_to(root / "nested" / "deep", target_is_directory=True)
    (root / "dir-link-outside").symlink_to(outside, target_is_directory=True)
    package_root = root.resolve()

    slow = _lookup(lambda: package_module._relative_file(package_root, value))
    fast = _lookup(lambda: package_module._inventoried_file(package_root, value, {}))

    if isinstance(slow, str):
        assert fast == slow
    else:
        assert fast[0] == slow
        assert fast[1].st_ino == slow.stat().st_ino


def test_the_inventory_lookup_resolves_each_directory_once(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "package"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / name).write_text(name, encoding="utf-8")
    package_root = root.resolve()
    asked = []
    realpath = package_module.os.path.realpath
    monkeypatch.setattr(package_module.os.path, "realpath",
                        lambda path, **kwargs: asked.append(path) or realpath(path, **kwargs))
    directories: dict[str, str] = {}

    for name in ("a", "b", "c"):
        package_module._inventoried_file(package_root, name, directories)

    assert asked == [str(package_root)]
