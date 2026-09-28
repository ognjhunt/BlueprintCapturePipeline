from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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
