"""Many immutable scene profiles share one worker service configuration."""

import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV, PROFILE_SCHEMA
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import SceneConfigurationSubmissionError
from blueprint_pipeline.task_evaluation_sam31_profile_registry import (
    REGISTRY_ENV, register_sam31_profile, resolve_sam31_profile,
)


def profile(root, name, commit):
    value = {"schema_version": PROFILE_SCHEMA, "source_commit": commit * 40, "scene": name}
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    path = root / (name + ".json")
    path.write_text(json.dumps(value))
    return path


def test_two_scenes_resolve_without_environment_switching(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    first = register_sam31_profile(profile_path=profile(tmp_path, "first", "a"), registry_root=registry)
    second = register_sam31_profile(profile_path=profile(tmp_path, "second", "b"), registry_root=registry)
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    assert str(resolve_sam31_profile({"server_profile_sha256": first["sha256"]})) == first["path"]
    assert str(resolve_sam31_profile({"server_profile_sha256": second["sha256"]})) == second["path"]
    assert first["source_commit"] != second["source_commit"]


def test_registration_is_idempotent_and_keeps_exact_bytes(tmp_path):
    source = profile(tmp_path, "first", "a")
    first = register_sam31_profile(profile_path=source, registry_root=tmp_path / "registry")
    assert register_sam31_profile(profile_path=source, registry_root=tmp_path / "registry") == first
    from pathlib import Path
    assert Path(first["path"]).read_bytes() == source.read_bytes()


def test_registry_cannot_follow_symlink_or_caller_path(tmp_path, monkeypatch):
    source = profile(tmp_path, "first", "a")
    record = register_sam31_profile(profile_path=source, registry_root=tmp_path / "registry")
    from pathlib import Path
    target = Path(record["path"])
    target.unlink()
    target.symlink_to(source)
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "registry"))
    with pytest.raises(SceneConfigurationSubmissionError, match="symlink"):
        resolve_sam31_profile({"server_profile_sha256": record["sha256"]})
    with pytest.raises(SceneConfigurationSubmissionError, match="digest_invalid"):
        resolve_sam31_profile({"server_profile_sha256": "../../secret"})


def test_legacy_profile_still_requires_exact_plan_digest(tmp_path, monkeypatch):
    source = profile(tmp_path, "first", "a")
    record = register_sam31_profile(profile_path=source, registry_root=tmp_path / "registry")
    monkeypatch.delenv(REGISTRY_ENV, raising=False)
    monkeypatch.setenv(PROFILE_ENV, str(source))
    assert resolve_sam31_profile({"server_profile_sha256": record["sha256"]}) == source
    with pytest.raises(SceneConfigurationSubmissionError, match="changed"):
        resolve_sam31_profile({"server_profile_sha256": "sha256:" + "f" * 64})


def test_preflight_accepts_content_registry_without_per_scene_file(tmp_path, monkeypatch):
    import os
    from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight

    registry = tmp_path / "registry"
    registry.mkdir()
    units = {name: {"effective_environment": {REGISTRY_ENV: str(registry)}} for name in (
        "blueprint-task-evaluation-launch-preparation.service",
        "blueprint-task-evaluation-sam31-preparation-execution.service")}
    findings = preflight.binding_checks(units, "a" * 40, (os.getuid(), os.getgid()))
    assert not any(row["code"] == "sam31_profile_unbound" for row in findings)
    assert sum(row["code"] == "sam31_profile_registry_content_bound" for row in findings) == 2
