"""Default assets share byte/closure verification, without live runtime claims."""
from __future__ import annotations

import copy
import hashlib
import os

import pytest

from blueprint_pipeline.blueprint_robot_model_delivery import (
    CONFIG_SCHEMA, BlueprintRobotModelError, require_default_robot_model, stage_default_robot_model,
)
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from tests.test_team_prepared_robot_model_delivery import setup_closure


def fixture(tmp_path, monkeypatch):
    source, closure = setup_closure(tmp_path)
    model = next(r for r in closure["files"] if r["path"] == closure["model_path"])
    imported = {"model": model, "dependency_closure": closure, "import_digest": "sha256:" + "e" * 64}
    monkeypatch.setattr("blueprint_pipeline.deployment_embodiment_asset_import.verify_prepared_asset_directory",
                        lambda path: copy.deepcopy(imported))
    config = {"schema_version": CONFIG_SCHEMA, "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID,
              "directory": str(source), "model_sha256": model["sha256"],
              "dependency_closure_digest": closure["closure_digest"]}
    return {"default_robot_model": config}, source, closure


def test_default_model_staging_reuses_closed_asset_and_survives_restart(tmp_path, monkeypatch):
    profile, source, closure = fixture(tmp_path, monkeypatch)
    target = tmp_path / "runtime"
    receipt = stage_default_robot_model(profile=profile, target_directory=target)
    assert receipt["dependency_closure"] == closure
    assert "directory" not in receipt
    assert receipt["execution_authorized"] is False
    assert stage_default_robot_model(profile=profile, target_directory=target) == receipt
    for row in closure["files"]:
        assert (target / row["path"]).read_bytes() == (source / row["path"]).read_bytes()


@pytest.mark.parametrize("which", ["root", "dependency", "symlink", "missing", "fence"])
def test_changed_dependency_or_root_refuses_at_execution_and_closeout(tmp_path, monkeypatch, which):
    profile, _, closure = fixture(tmp_path, monkeypatch)
    target = tmp_path / "runtime"
    stage_default_robot_model(profile=profile, target_directory=target)
    path = target / ("robot.usda" if which == "root" else "textures/body.png")
    if which in ("root", "dependency"):
        path.write_bytes(b"changed")
    elif which in ("symlink", "missing"):
        path.unlink()
        if which == "symlink":
            path.symlink_to(target / "robot.usda")
    with pytest.raises(BlueprintRobotModelError):
        require_default_robot_model(profile=profile, model_path=target / "robot.usda",
            expected_model_sha256=profile["default_robot_model"]["model_sha256"],
            expected_closure_digest="sha256:" + "f" * 64 if which == "fence" else closure["closure_digest"])


def test_operator_binding_cannot_be_derived_from_job_payload():
    from blueprint_pipeline.team_candidate_quick10_registry import DERIVED_FIELDS, OPERATOR_FIELDS

    for field in ("default_robot_model", "site_task_qualification"):
        assert field in OPERATOR_FIELDS
        assert field not in DERIVED_FIELDS


def test_qualification_pins_default_closure_and_verifies_each_dependency(tmp_path, monkeypatch):
    from blueprint_pipeline.site_task_simulation_qualification import qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW, _reauthorize_review, _rewrite, fixture as qualification

    evidence = tmp_path / "evidence"
    profile = qualification(evidence, binding={"policy_interface_id": "openpi_droid.v1"})
    staging, source, closure = fixture(tmp_path, monkeypatch)
    # The synthetic operator asset and synthetic qualification use the same exact model bytes.
    model = (evidence / "robot_model.json").read_bytes()
    (source / "robot.usda").write_bytes(model)
    closure["files"][0].update(sha256="sha256:" + hashlib.sha256(model).hexdigest(), size_bytes=len(model))
    closure["closure_digest"] = digest(closure, digest_field="closure_digest")
    staging["default_robot_model"].update(model_sha256=closure["files"][0]["sha256"],
        dependency_closure_digest=closure["closure_digest"])
    imported = {"model": closure["files"][0], "dependency_closure": closure, "import_digest": "sha256:" + "e" * 64}
    monkeypatch.setattr("blueprint_pipeline.deployment_embodiment_asset_import.verify_prepared_asset_directory",
                        lambda path: copy.deepcopy(imported))
    profile.update(staging)
    profile["site_task_qualification"]["default_robot_model_closure_digest"] = closure["closure_digest"]
    stage_default_robot_model(profile=profile, target_directory=evidence / "robot")
    _rewrite(profile, "request", lambda r: next(a for a in r["source_artifacts"] if a["role"] == "robot_model")
             .update(path="robot/robot.usda"))
    _reauthorize_review(profile)
    receipt = qualify_from_profile(profile, now=NOW)
    assert receipt["binding"]["robot_model_digest"] == staging["default_robot_model"]["model_sha256"]
    (evidence / "robot" / "textures" / "body.png").write_bytes(b"unreviewed-mesh-or-texture")
    with pytest.raises(BlueprintRobotModelError, match="runtime_bytes_changed"):
        qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("mode", [0o777, 0o775])
def test_writable_operator_source_is_refused(tmp_path, monkeypatch, mode):
    profile, source, _ = fixture(tmp_path, monkeypatch)
    os.chmod(source, mode)
    with pytest.raises(BlueprintRobotModelError, match="prepared_asset_invalid"):
        stage_default_robot_model(profile=profile, target_directory=tmp_path / "runtime")


def test_an_explicit_team_robot_does_not_inherit_default_asset_selection(tmp_path):
    from blueprint_pipeline.site_task_simulation_qualification import qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW, fixture as qualification

    profile = qualification(tmp_path)
    original = qualify_from_profile(profile, now=NOW)
    profile["default_robot_model"] = {"unrelated": "operator default for Blueprint DROID runs"}
    assert qualify_from_profile(profile, now=NOW) == original
