"""Synthetic engine/render records test extraction, never site qualification."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.site_task_simulator_measurements import (
    CLOSED_SNAPSHOT_SCHEMA, RECIPE_SCHEMA, SNAPSHOT_SCHEMA, SimulatorMeasurementError,
    produce_from_operator_profile, produce_measurements,
    retain_measurements,
)
from tests.test_site_task_simulation_qualification import fixture as qualification_fixture


def _record(path):
    raw = path.read_bytes()
    return {"path": path.name, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def fixture(root, *, binding=None):
    profile = qualification_fixture(root, binding=binding)
    binding = json.loads((root / "request.json").read_text())["binding"]
    (root / "scene.json").write_text(json.dumps({"scene_revision_digest": binding["scene_revision_digest"]}))
    rgb = np.zeros((21, 21, 3), dtype=np.uint8)
    np.savez(root / "render.npz", rgb=rgb, depth_m=np.full((21, 21), 2.0),
             intrinsics=np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1]]))
    snapshot = {"schema_version": SNAPSHOT_SCHEMA, "binding": binding,
        "source_kind": "native_engine_readback", "engine_build": "synthetic-test-engine",
        "runtime_digest": "sha256:" + "b" * 64, "observed_at_iso": "2026-10-17T10:01:00Z",
        "scene_artifact": _record(root / "scene.json"), "robot_model_artifact": _record(root / "robot_model.json"),
        "geometry_points_m": {"pick": [0, 0, .7], "place": [.4, 0, .7]},
        "registered_floor_plane": [0, 0, 1, 0],
        "collision_bounds_m": {"object": {"min": [0, 0, 0], "max": [.2, .1, .1]}},
        "collision_rays": {"front": {"query": "nearest_scene_collision", "hit": True,
                                      "origin_m": [0, 0, 0], "hit_point_m": [.5, 0, 0]}},
        "rigid_body_masses_kg": {"object": 2.0}, "calibration_sample_ids": [],
        "rendered_rgbd": {"policy-view": {"artifact": _record(root / "render.npz"),
            "model": "rectified_pinhole", "depth_quantity": "optical_axis_m",
            "renderer_digest": "sha256:" + "c" * 64, "camera_pose_digest": "sha256:" + "d" * 64}}}
    snapshot["snapshot_digest"] = digest(snapshot, digest_field="snapshot_digest")
    (root / "snapshot.json").write_text(json.dumps(snapshot))
    readings = [
        {"check_id": "pick_place.scale.lengths", "sample_id": "pick_to_place_distance",
         "operation": "horizontal_distance", "points": ["pick", "place"]},
        {"check_id": "pick_place.registration.surface_heights", "sample_id": "pick_surface_height",
         "operation": "floor_height", "name": "pick"},
        {"check_id": "pick_place.collision.clearances", "sample_id": "pick_front_clearance",
         "operation": "collision_ray_clearance", "name": "front"},
        *[{"check_id": "pick_place.task_state.object_size", "sample_id": "object_" + axis,
           "operation": "collision_extent", "name": "object", "axis": i}
          for i, axis in enumerate(("length", "width", "height"))],
        {"check_id": "pick_place.observation.visible_lengths", "sample_id": "object_length",
         "operation": "rgbd_length", "name": "policy-view", "pixels": [[5, 10], [15, 10]]},
        {"check_id": "pick_place.task_dynamics.object_mass", "sample_id": "object_mass",
         "operation": "rigid_body_mass", "name": "object"}]
    recipe = {"schema_version": RECIPE_SCHEMA, "family": "rigid_pick_place", "binding": binding,
              "producer_id": "test-simulator", "snapshot": _record(root / "snapshot.json"), "readings": readings}
    return recipe, snapshot, profile


def _rewrite_snapshot(root, recipe, snapshot):
    snapshot["snapshot_digest"] = digest(snapshot, digest_field="snapshot_digest")
    (root / "snapshot.json").write_text(json.dumps(snapshot))
    recipe["snapshot"] = _record(root / "snapshot.json")


def _operator_recipe(root, recipe, profile):
    path = root / "recipe.json"
    path.write_text(json.dumps(recipe))
    profile["simulator_measurement_producer"] = {
        "recipe_path": str(path), "recipe_digest": digest(recipe),
        "evidence_root": str(root), "producer_id": recipe["producer_id"]}
    return profile


def test_root_model_only_readings_cannot_enter_operator_qualification_preparation(tmp_path):
    recipe, _, profile = fixture(tmp_path)
    _operator_recipe(tmp_path, recipe, profile)
    # The original retained format still supports diagnostic extraction, but
    # its root-file hash cannot stand in for an authenticated robot closure.
    diagnostic = produce_measurements(recipe, evidence_root=tmp_path)
    assert diagnostic["checks"]
    assert diagnostic["extraction"]["robot_model_dependency_evidence"] == {
        "status": "unproven", "runtime_open_proven": False}
    with pytest.raises(SimulatorMeasurementError, match="robot_model_closure_unproven"):
        produce_from_operator_profile(profile)


def _closed_fixture(root, monkeypatch):
    from blueprint_pipeline.blueprint_robot_model_delivery import stage_default_robot_model
    from tests.test_blueprint_robot_model_delivery import fixture as model_fixture
    from tests.test_site_task_simulation_qualification import _rewrite

    recipe, snapshot, profile = fixture(root, binding={"policy_interface_id": "openpi_droid.v1"})
    operator = root / "operator"
    operator.mkdir()
    model_profile, source, closure = model_fixture(operator, monkeypatch)
    # Explicit synthetic import; this exercises byte coupling, never USD load,
    # engine provenance, measurements or a real qualification claim.
    model = (root / "robot_model.json").read_bytes()
    (source / closure["model_path"]).write_bytes(model)
    selected = next(row for row in closure["files"] if row["path"] == closure["model_path"])
    selected.update(sha256="sha256:" + hashlib.sha256(model).hexdigest(), size_bytes=len(model))
    closure["closure_digest"] = digest(closure, digest_field="closure_digest")
    model_profile["default_robot_model"].update(model_sha256=selected["sha256"],
        dependency_closure_digest=closure["closure_digest"])
    imported = {"model": selected, "dependency_closure": closure, "import_digest": "sha256:" + "e" * 64}
    monkeypatch.setattr("blueprint_pipeline.deployment_embodiment_asset_import.verify_prepared_asset_directory",
                        lambda path: copy.deepcopy(imported))
    profile.update(model_profile)
    profile["site_task_qualification"]["default_robot_model_closure_digest"] = closure["closure_digest"]
    runtime = root / "runtime-robot"
    stage_default_robot_model(profile=profile, target_directory=runtime)
    model_record = {**_record(runtime / closure["model_path"]), "path": "runtime-robot/" + closure["model_path"]}
    snapshot.update(schema_version=CLOSED_SNAPSHOT_SCHEMA, robot_model_artifact=model_record,
        robot_model_dependency_closure_digest=closure["closure_digest"], robot_model_source_import_digest=imported["import_digest"])
    _rewrite_snapshot(root, recipe, snapshot)
    _rewrite(profile, "request", lambda request: next(row for row in request["source_artifacts"]
        if row["role"] == "robot_model").update(model_record))
    _operator_recipe(root, recipe, profile)
    return recipe, snapshot, profile, imported


def test_operator_preparation_binds_all_retained_model_bytes_and_source_generation(tmp_path, monkeypatch):
    _, _, profile, imported = _closed_fixture(tmp_path, monkeypatch)
    document = produce_from_operator_profile(profile)
    evidence = document["extraction"]["robot_model_dependency_evidence"]
    assert evidence["dependency_closure"] == imported["dependency_closure"]
    assert evidence["source_import_digest"] == imported["import_digest"]
    assert evidence["status"] == "retained_bytes_verified" and evidence["runtime_open_proven"] is False
    assert document["extraction"]["method_id"].endswith(".v2")
    names = {row["path"] for row in document["extraction"]["source_artifacts"]}
    assert {"runtime-robot/robot.usda", "runtime-robot/textures/body.png"} <= names
    assert produce_from_operator_profile(profile) == document


@pytest.mark.parametrize("change", ["dependency", "root", "missing", "symlink", "source_generation",
    "snapshot_closure", "review_closure", "missing_closure", "root_path", "operator_binding", "config_closure"])
def test_closed_metrology_refuses_changed_dependencies_and_generations(tmp_path, monkeypatch, change):
    recipe, snapshot, profile, imported = _closed_fixture(tmp_path, monkeypatch)
    model = tmp_path / "runtime-robot" / "robot.usda"
    dependency = model.parent / "textures" / "body.png"
    if change in {"dependency", "root"}:
        (dependency if change == "dependency" else model).write_bytes(b"changed retained bytes")
    elif change in {"missing", "symlink"}:
        dependency.unlink()
        if change == "symlink":
            dependency.symlink_to(model)
    elif change == "source_generation":
        imported["import_digest"] = "sha256:" + "f" * 64
    elif change == "snapshot_closure":
        snapshot["robot_model_dependency_closure_digest"] = "sha256:" + "f" * 64
    elif change == "review_closure":
        profile["site_task_qualification"]["default_robot_model_closure_digest"] = "sha256:" + "f" * 64
    elif change == "missing_closure":
        snapshot.pop("robot_model_dependency_closure_digest")
    elif change == "operator_binding":
        profile.pop("default_robot_model")
    elif change == "config_closure":
        profile["default_robot_model"]["dependency_closure_digest"] = "sha256:" + "f" * 64
    else:
        alternate = tmp_path / "renamed.usda"
        alternate.write_bytes(model.read_bytes())
        snapshot["robot_model_artifact"] = _record(alternate)
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    _operator_recipe(tmp_path, recipe, profile)
    with pytest.raises(SimulatorMeasurementError):
        produce_from_operator_profile(profile)


def test_closed_snapshot_cannot_use_job_or_snapshot_bytes_as_operator_authority(tmp_path, monkeypatch):
    recipe, _, _, _ = _closed_fixture(tmp_path, monkeypatch)
    with pytest.raises(SimulatorMeasurementError, match="robot_model_operator_required"):
        produce_measurements(recipe, evidence_root=tmp_path)


def test_six_dimensions_calculated_from_distinct_retained_sources(tmp_path):
    recipe, _, _ = fixture(tmp_path)
    document = produce_measurements(recipe, evidence_root=tmp_path)
    values = {row["dimension"]: [s["value"] for s in row["samples"]] for row in document["checks"]}
    assert values == {"scale": [.4], "registration": [.7], "collision": [.5],
                      "task_state": [.2, .1, .1], "observation": [.2], "task_dynamics": [2.]}
    assert document["physical_performance_claimed"] is False
    provenance = {r["provenance"] for r in document["extraction"]["samples"]}
    assert {"calibrated_rendered_rgbd", "engine_collision_geometry", "engine_rigid_body_attribute"} <= provenance
    assert len(document["extraction"]["source_artifacts"]) == 4


def test_appearance_misalignment_is_observed_even_with_matching_colliders(tmp_path):
    recipe, snapshot, _ = fixture(tmp_path)
    initial = produce_measurements(recipe, evidence_root=tmp_path)
    with np.load(tmp_path / "render.npz") as arrays:
        rgb, depth, k = arrays["rgb"], arrays["depth_m"], arrays["intrinsics"]
    np.savez(tmp_path / "render.npz", rgb=rgb, depth_m=depth * 2, intrinsics=k)
    snapshot["rendered_rgbd"]["policy-view"]["artifact"] = _record(tmp_path / "render.npz")
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    changed = produce_measurements(recipe, evidence_root=tmp_path)
    before, after = ({c["dimension"]: c["samples"] for c in d["checks"]} for d in (initial, changed))
    assert after["task_state"] == before["task_state"]
    assert after["observation"][0]["value"] == pytest.approx(.4)


def test_mass_readback_changes_dynamics_without_changing_geometry(tmp_path):
    recipe, snapshot, _ = fixture(tmp_path)
    snapshot["rigid_body_masses_kg"]["object"] = 20
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    doc = produce_measurements(recipe, evidence_root=tmp_path)
    assert next(c for c in doc["checks"] if c["dimension"] == "task_dynamics")["samples"][0]["value"] == 20
    assert next(c for c in doc["checks"] if c["dimension"] == "task_state")["samples"][0]["value"] == .2


@pytest.mark.parametrize("failure,reason", [
    ("calibration", "calibration_reading_reused"), ("scene", "snapshot_binding_invalid"),
    ("missing", "coverage_incomplete"), ("duplicate", "reading_binding_invalid"),
    ("fake_observation", "reading_provenance_invalid"), ("unit", "render_calibration_invalid"),
    ("model", "robot_model_binding_invalid"), ("unknown_family", "recipe_or_family_invalid")])
def test_missing_or_mismatched_evidence_refuses(tmp_path, failure, reason):
    recipe, snapshot, _ = fixture(tmp_path)
    if failure == "calibration":
        snapshot["calibration_sample_ids"] = ["pick_to_place_distance"]
    elif failure == "scene":
        snapshot["binding"] = {**snapshot["binding"], "capture_id": "other-capture"}
    elif failure == "missing":
        recipe["readings"].pop()
    elif failure == "duplicate":
        recipe["readings"].append(copy.deepcopy(recipe["readings"][0]))
    elif failure == "fake_observation":
        recipe["readings"][-2].update(operation="collision_extent", name="object", axis=0)
    elif failure == "unit":
        snapshot["rendered_rgbd"]["policy-view"]["depth_quantity"] = "millimetres"
    elif failure == "model":
        (tmp_path / "robot_model.json").write_bytes(b"changed")
        snapshot["robot_model_artifact"] = _record(tmp_path / "robot_model.json")
    elif failure == "unknown_family":
        recipe["family"] = "pouring"
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    with pytest.raises(SimulatorMeasurementError, match=reason):
        produce_measurements(recipe, evidence_root=tmp_path)


def test_source_mutation_and_symlink_refuse(tmp_path):
    recipe, _, _ = fixture(tmp_path)
    (tmp_path / "render.npz").write_bytes(b"changed")
    with pytest.raises(SimulatorMeasurementError, match="source_bytes_changed"):
        produce_measurements(recipe, evidence_root=tmp_path)
    (tmp_path / "snapshot.json").rename(tmp_path / "actual.json")
    (tmp_path / "snapshot.json").symlink_to(tmp_path / "actual.json")
    with pytest.raises(SimulatorMeasurementError, match="source_path_invalid"):
        produce_measurements(recipe, evidence_root=tmp_path)


def test_restart_adopts_byte_identical_document_without_replacing_output(tmp_path):
    recipe, _, _ = fixture(tmp_path)
    first = produce_measurements(recipe, evidence_root=tmp_path)
    path = tmp_path / "simulator-output.json"
    retain_measurements(path, first)
    original = path.read_bytes()
    retain_measurements(path, produce_measurements(recipe, evidence_root=tmp_path))
    assert path.read_bytes() == original
    with pytest.raises(SimulatorMeasurementError, match="output_conflict"):
        retain_measurements(path, {**first, "producer_id": "another"})
    assert path.read_bytes() == original


def test_existing_qualification_command_prepares_only_operator_pinned_readings(tmp_path, monkeypatch):
    from blueprint_pipeline.site_task_simulation_qualification import main
    from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    recipe, _, selected_profile, _ = _closed_fixture(tmp_path / "evidence", monkeypatch)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    profile = {"capture_root": str(tmp_path.resolve()), "default_robot_model": selected_profile["default_robot_model"],
        "site_task_qualification": selected_profile["site_task_qualification"], "simulator_measurement_producer": {
        "recipe_path": str(recipe_path), "recipe_digest": digest(recipe),
        "evidence_root": str(tmp_path / "evidence"), "producer_id": recipe["producer_id"]}}
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [profile]}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(path))
    out = tmp_path / "output.json"
    args = ["--capture-root", str(tmp_path), "--out", str(out), "--prepare-simulator-measurements"]
    assert main(args) == 0
    assert json.loads(out.read_text())["schema_version"] == "blueprint.site_task_measurement_samples.v1"
    original = out.read_bytes()
    assert main(args) == 0 and out.read_bytes() == original
    recipe_path.write_text("{}")
    with pytest.raises(SimulatorMeasurementError, match="operator_recipe_changed"):
        main(args)


def _paired_qualification(root, monkeypatch):
    from blueprint_pipeline.site_task_fidelity_profile_templates import template
    from tests.test_site_task_simulation_qualification import _reauthorize_review, _rewrite

    recipe, snapshot, profile, _ = _closed_fixture(root, monkeypatch)
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    profile["simulator_measurement_producer"] = {
        "recipe_path": str(recipe_path), "recipe_digest": digest(recipe), "evidence_root": str(root),
        "producer_id": recipe["producer_id"]}
    generated = produce_from_operator_profile(profile)
    (root / "simulator_measurement.json").write_text(json.dumps(generated))
    criteria = [{k: row[k] for k in ("id", "dimension", "unit", "minimum_sample_count", "maximum_absolute_error", "maximum_rms_error")}
                for row in template("rigid_pick_place")["checks"]]
    _rewrite(profile, "profile", lambda p: p.update(checks=criteria))
    # Independent synthetic site readings are explicit controls. No production
    # path copies simulator numbers into independent measurements.
    expected = {"pick_to_place_distance": .4, "pick_surface_height": .7, "pick_front_clearance": .5,
                "object_length": .2, "object_width": .1, "object_height": .1, "object_mass": 2.0}
    observed = json.loads((root / "measurement.json").read_text())
    observed["checks"] = [{"id": row["id"], "dimension": row["dimension"], "unit": row["unit"],
                           "samples": [{"sample_id": s["sample_id"], "value": expected[s["sample_id"]]}
                                       for s in row["samples"]]} for row in generated["checks"]]
    (root / "measurement.json").write_text(json.dumps(observed))
    def sources(request):
        for source in request["source_artifacts"]:
            if source["role"] in {"measurement", "simulator_measurement", "scene"}:
                source.update({k: v for k, v in _record(root / source["path"]).items() if k != "path"})
    _rewrite(profile, "request", sources)
    _reauthorize_review(profile)
    profile["site_task_qualification"]["allowed_simulator_producers"] = [recipe["producer_id"]]
    return recipe, snapshot, profile


def test_qualifier_recomputes_generated_readings_from_original_runtime_sources(tmp_path, monkeypatch):
    from blueprint_pipeline.site_task_simulation_qualification import qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW

    _, _, profile = _paired_qualification(tmp_path, monkeypatch)
    receipt = qualify_from_profile(profile, now=NOW)
    assert all(c["max_abs_error"] == 0 for field in ("registration_receipt", "fidelity_review_receipt")
               for c in receipt[field]["checks"])
    (tmp_path / "render.npz").write_bytes(b"changed without a new review")
    with pytest.raises(SimulatorMeasurementError, match="source_bytes_changed"):
        qualify_from_profile(profile, now=NOW)


def test_qualifier_does_not_fall_back_to_root_only_metrology(tmp_path, monkeypatch):
    from blueprint_pipeline.site_task_simulation_qualification import qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW, _reauthorize_review, _rewrite

    recipe, snapshot, profile = _paired_qualification(tmp_path, monkeypatch)
    assert qualify_from_profile(profile, now=NOW)["receipt_digest"]
    snapshot["schema_version"] = SNAPSHOT_SCHEMA
    snapshot.pop("robot_model_dependency_closure_digest")
    snapshot.pop("robot_model_source_import_digest")
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    _operator_recipe(tmp_path, recipe, profile)
    diagnostic = produce_measurements(recipe, evidence_root=tmp_path)
    (tmp_path / "simulator_measurement.json").write_text(json.dumps(diagnostic))
    _rewrite(profile, "request", lambda request: next(row for row in request["source_artifacts"]
        if row["role"] == "simulator_measurement").update(_record(tmp_path / "simulator_measurement.json")))
    _reauthorize_review(profile)
    with pytest.raises(SimulatorMeasurementError, match="robot_model_closure_unproven"):
        qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("dimension", ["observation", "task_dynamics"])
def test_wrong_appearance_or_mass_fails_existing_fidelity_pairing(tmp_path, monkeypatch, dimension):
    from blueprint_pipeline.site_task_simulation_qualification import SiteTaskQualificationError, qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW, _reauthorize_review, _rewrite

    recipe, snapshot, profile = _paired_qualification(tmp_path, monkeypatch)
    if dimension == "observation":
        with np.load(tmp_path / "render.npz") as arrays:
            rgb, depth, k = arrays["rgb"], arrays["depth_m"], arrays["intrinsics"]
        np.savez(tmp_path / "render.npz", rgb=rgb, depth_m=depth * 2, intrinsics=k)
        snapshot["rendered_rgbd"]["policy-view"]["artifact"] = _record(tmp_path / "render.npz")
    else:
        snapshot["rigid_body_masses_kg"]["object"] = 20
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    profile["simulator_measurement_producer"]["recipe_digest"] = digest(recipe)
    (tmp_path / "simulator_measurement.json").write_text(json.dumps(produce_from_operator_profile(profile)))
    def source(request):
        next(a for a in request["source_artifacts"] if a["role"] == "simulator_measurement").update(
            {k: v for k, v in _record(tmp_path / "simulator_measurement.json").items() if k != "path"})
    _rewrite(profile, "request", source)
    _reauthorize_review(profile)
    with pytest.raises(SiteTaskQualificationError, match="fidelity_failed:pick_place." + dimension):
        qualify_from_profile(profile, now=NOW)


@pytest.mark.parametrize("changed_bytes", [False, True])
def test_metrology_scene_must_join_the_reviewed_scene_bytes_not_only_revision(tmp_path, monkeypatch, changed_bytes):
    from blueprint_pipeline.site_task_simulation_qualification import SiteTaskQualificationError, qualify_from_profile
    from tests.test_site_task_simulation_qualification import NOW, _reauthorize_review, _rewrite

    recipe, snapshot, profile = _paired_qualification(tmp_path, monkeypatch)
    alternate = tmp_path / "different-scene.json"
    original = (tmp_path / "scene.json").read_bytes()
    if changed_bytes:
        scene = json.loads(original)
        scene["unreviewed_geometry"] = {"support_height_m": 100}
        alternate.write_text(json.dumps(scene))
    else:
        alternate.write_bytes(original)
    snapshot["scene_artifact"] = _record(alternate)
    _rewrite_snapshot(tmp_path, recipe, snapshot)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    profile["simulator_measurement_producer"]["recipe_digest"] = digest(recipe)
    (tmp_path / "simulator_measurement.json").write_text(json.dumps(produce_from_operator_profile(profile)))
    def source(request):
        next(a for a in request["source_artifacts"] if a["role"] == "simulator_measurement").update(
            {k: v for k, v in _record(tmp_path / "simulator_measurement.json").items() if k != "path"})
    _rewrite(profile, "request", source)
    _reauthorize_review(profile)
    # The attacker re-pins the recipe/document/review but retains the scene
    # revision string. Those do not make different scene bytes applicable.
    if changed_bytes:
        with pytest.raises(SiteTaskQualificationError, match="simulator_scene_bytes_mismatch"):
            qualify_from_profile(profile, now=NOW)
    else:
        assert qualify_from_profile(profile, now=NOW)["evidence_scope"] == "simulation"
