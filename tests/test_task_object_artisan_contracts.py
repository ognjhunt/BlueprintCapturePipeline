"""ADP-009 source authority and new-request policy stay separate from history."""
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_backend import (
    ARTISAN_BUDGET_ENV, BACKEND_SETTING_ENV, backend_configuration, default_backend, frozen_authoring_cap,
)
from blueprint_pipeline.task_object_authoring_contracts import build_construction_request, verify_construction_request, file_record, AuthoringContractError
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import stage_three_configuration, spend_block
from blueprint_pipeline.task_evaluation_launch_preparation_contract import validate_launch_preparation_request


def source_request(tmp_path, *, frames=1):
    source = tmp_path / "source.obj"
    source.write_text("# retained source geometry bytes")
    record = file_record(source)
    record["digest"] = record.pop("sha256")
    paths = []
    for index in range(frames):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (4, 4), color=(index, 20, 30)).save(path)
        paths.append(path)
    configuration = {"schema_version": "rigid_replacement_authoring_configuration.v1", "scene_id": "fixture-scene",
        "authoring_backend": "artisan_v1", **backend_configuration("artisan_v1"), "physics_authority_granted_by_authoring": False,
        "replacement_identity": {"id": "replacement", "version": "v1"}, "source_object_identity": "observed-object",
        "metric_envelope": {"minimum_xyz_m": [1, 2, 3], "maximum_xyz_m": [1.2, 2.3, 3.4], "maximum_dimension_relative_error": 0.05},
        "dimension_authority": "estimated"}
    stage = {"configuration": configuration, "configuration_sha256": "sha256:" + "a" * 64,
             "source_commit": "b" * 40, "run_id": "fixture-run", "construction_envelope": {}}
    deps = [{"output_artifacts": [{"role": "source_object_candidate_mesh", **record}]}]
    return stage, deps, paths


def test_all_admitted_views_and_source_uncertainty_survive_without_global_count_limit(tmp_path):
    stage, dependencies, frames = source_request(tmp_path, frames=21)
    request = build_construction_request(stage, dependencies, frames, {"role": "original_capture"})
    assert len(request["source_frames"]) == 21 and request["view_selection"]["omitted"] == []
    assert request["dimensions_m"] == pytest.approx([0.2, 0.3, 0.4])
    assert request["dimension_uncertainty_m"] == pytest.approx([0.01, 0.015, 0.02])
    assert request["maximum_export_error_m"] == 0.001 and request["dimension_authority"] == "estimated"
    verify_construction_request(request)
    frames[0].write_bytes(b"changed observed source")
    with pytest.raises(AuthoringContractError, match="source_bytes_changed"):
        verify_construction_request(request)


def test_source_dimensions_in_semantic_body_frame_are_not_rotated_world_aabb(tmp_path):
    stage, deps, frames = source_request(tmp_path)
    stage["configuration"]["body_extent_m"] = {"depth": 0.6, "width": 0.5, "height": 1.0, "frame": "assembly_front_+X_up_Z", "basis": "estimated"}
    request = build_construction_request(stage, deps, frames, {})
    assert request["dimensions_m"] == [0.6, 0.5, 1]
    assert request["configuration"]["metric_envelope"] == stage["configuration"]["metric_envelope"]


@pytest.mark.parametrize("edge,axis", [("bottom", [0, 1, 0]), ("left", [0, 0, -1])])
def test_hinge_axis_is_distinct_from_pull_direction(tmp_path, edge, axis):
    stage, deps, frames = source_request(tmp_path)
    stage["configuration"].update(hinge_edge=edge, mechanism={"joint_type": "revolute", "opening_axis_asset_frame": [1, 0, 0]})
    request = build_construction_request(stage, deps, frames, {})
    assert request["joint_constraints"]["physical_joint_axis_asset_frame"] == axis
    assert request["joint_constraints"]["pull_direction_asset_frame"] == [1, 0, 0]


def test_unverified_measurement_cannot_be_promoted_by_a_label(tmp_path):
    stage, deps, frames = source_request(tmp_path)
    stage["configuration"]["measured_physical_properties"] = {"mass_kg": {"value": 2, "basis": "measured",
        "interval": {"lower": 2, "upper": 2}, "rationale": "scale", "uncertainty": "unknown", "evidence_ids": ["missing"]}}
    with pytest.raises(AuthoringContractError, match="measured_property_evidence_required"):
        build_construction_request(stage, deps, frames, {})


def test_editable_budget_freezes_into_quote_and_intake_above_historical_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv(ARTISAN_BUDGET_ENV, "80")
    stage, _, _ = source_request(tmp_path)
    frozen = stage["configuration"]
    quote = spend_block("artisan_v1", authoring_max_cost_usd=frozen_authoring_cap(frozen))
    assert quote["hard_cap_usd"] == 91.76
    path = Path(__file__).parent / "fixtures/scene_configuration/astra_preparation_request.v1.json"
    intake = json.loads(path.read_text())
    intake.update(replacement_authoring_backend="artisan_v1", spend=quote)
    assert validate_launch_preparation_request(intake) == intake
    monkeypatch.setenv(ARTISAN_BUDGET_ENV, "5")
    assert frozen_authoring_cap(frozen) == 80
    assert backend_configuration("artisan_v1")["authoring_budget"]["maximum_cost_usd"] == 5


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "uncapped"])
def test_unsupported_uncapped_or_nonfinite_budget_refuses(value, monkeypatch):
    monkeypatch.setenv(ARTISAN_BUDGET_ENV, value)
    with pytest.raises(ValueError, match="budget_invalid"):
        backend_configuration("artisan_v1")


def test_new_default_dispatch_imports_no_legacy_authoring_engine(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv(BACKEND_SETTING_ENV, "artisan_v1")  # switched on (go-live)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"configuration": {"authoring_backend": "artisan_v1"}}))
    code = '''import os,sys
from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver
calls=[]
driver.execute_artisan_component=lambda: calls.append('artisan')
from blueprint_pipeline.task_evaluation_scene_configuration_authoring_driver import main
assert main()==0 and calls==['artisan']
assert not any('task_object_astra_authoring' in name or 'content_agents_driver' in name for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ,
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    config = stage_three_configuration(scene_id="scene", replacement_identity={"id": "asset", "version": "v1"},
        source_instance_id="object", authoring_target="source object", source_min=[0, 0, 0], source_max=[1, 1, 1],
        dimension_tolerance=0.1, physics_bounds={"mass_kg_bounds": [1, 3], "static_friction_bounds": [0.3, 0.6],
            "dynamic_friction_bounds": [0.1, 0.3], "restitution_bounds": [0, 0.1]})
    assert config["authoring_backend"] == "artisan_v1"
    assert canonical_digest(config) != canonical_digest({**config, "authoring_backend": "astra_cad_blender_v1"})


def test_one_setting_selects_the_builder_and_artisan_starts_switched_off(monkeypatch):
    """Owner decision 2026-10-13: ARTISAN is merged but off until its runtime is installed."""
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
        articulated_stage_three_configuration,
    )
    from tests.test_task_object_articulated_packaging import MECHANISM, PHYSICS

    def rigid(**kwargs):
        return stage_three_configuration(scene_id="scene", replacement_identity={"id": "asset", "version": "v1"},
            source_instance_id="object", authoring_target="source object", source_min=[0, 0, 0],
            source_max=[1, 1, 1], dimension_tolerance=0.1, physics_bounds={"mass_kg_bounds": [1, 3],
                "static_friction_bounds": [0.3, 0.6], "dynamic_friction_bounds": [0.1, 0.3],
                "restitution_bounds": [0, 0.1]}, **kwargs)

    def articulated(**kwargs):
        return articulated_stage_three_configuration(scene_id="scene", replacement_identity={"id": "a", "version": "v1"},
            source_instance_id="cabinet", authoring_target="cabinet", source_min=[-0.21, -0.275, 0.0],
            source_max=[0.21, 0.275, 0.62], dimension_tolerance=0.2, physics_bounds=PHYSICS, mechanism=MECHANISM,
            **kwargs)

    monkeypatch.delenv(BACKEND_SETTING_ENV, raising=False)
    assert default_backend() == "astra_cad_blender_v1"
    assert rigid()["authoring_backend"] == articulated()["authoring_backend"] == "astra_cad_blender_v1"
    assert "authoring_budget" not in rigid() and "authoring_framework" not in articulated()
    monkeypatch.setenv(BACKEND_SETTING_ENV, "artisan_v1")
    assert rigid()["authoring_backend"] == articulated()["authoring_backend"] == "artisan_v1"
    assert rigid()["authoring_budget"]["mode"] == "finite_cap"
    # A named backend is never rewritten by the setting, in either direction.
    assert rigid(authoring_backend="astra_cad_blender_v1")["authoring_backend"] == "astra_cad_blender_v1"
    monkeypatch.delenv(BACKEND_SETTING_ENV)
    assert rigid(authoring_backend="artisan_v1")["authoring_backend"] == "artisan_v1"
    for value in ("content_agents", "artisan", "ASTRA"):
        with pytest.raises(ValueError, match="task_object_authoring_backend_setting_invalid"):
            default_backend({BACKEND_SETTING_ENV: value})


def test_deploy_and_release_provisioning_follow_the_one_setting(monkeypatch):
    """Unnamed, both select Astra; the ARTISAN runtime roots are required only once switched on."""
    import importlib.util
    root = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("provision_release_for_setting_test",
                                                  root / "provision_task_evaluation_scene_configuration_release.py")
    provision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provision)
    monkeypatch.delenv(BACKEND_SETTING_ENV, raising=False)
    calls = {"repository_root": Path(__file__).resolve().parents[1], "runtime_root": "/nonexistent-runtime-root"}
    # Switched off: an unnamed backend is Astra, which needs the legacy sources, not ARTISAN's.
    with pytest.raises(ValueError, match="scene_configuration_release_explicit_legacy_runtime_required"):
        provision.provision_scene_configuration_release(**calls, **_provision_placeholders())
    monkeypatch.setenv(BACKEND_SETTING_ENV, "artisan_v1")
    with pytest.raises(ValueError, match="scene_configuration_release_artisan_installed_runtime_required"):
        provision.provision_scene_configuration_release(**calls, **_provision_placeholders())
    deploy_source = (root / "deploy_control_plane_commit.py").read_text()
    assert 'authoring_backend: str | None = None' in deploy_source
    assert "authoring_backend = authoring_backend or default_backend()" in deploy_source


def _provision_placeholders():
    # The backend checks run before any of these are read.
    return {"source_commit": "0" * 40, "node_executable": "/nonexistent/node", "browser_root": "/nonexistent/browser",
            "browser_executable": "/nonexistent/browser/chrome", "node_modules_root": "/nonexistent/node_modules",
            "artifixer_root": "/nonexistent/artifixer", "readback": None, "readback_actor": "test"}
