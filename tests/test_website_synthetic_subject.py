"""A designed object a task adds: its start spot and its construction authoring."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import trimesh

from blueprint_pipeline.website_support_geometry import support_under
from blueprint_pipeline.website_synthetic_subject import choose_start, design_physics, write_candidate

BALL = {"semantic_label": "ball", "shape": "sphere", "dimensions_m": [0.065, 0.065, 0.065], "mass_kg": 0.06,
        "task_basis_quote": "a synthetic ball", "start_support_target_id": "floor", "decision_reason": "adds a ball",
        "design_basis": "synthetic_design_choice", "physical_measurement": False}


def _floor_with(*boxes):
    """A 4 m floor at z 0 (Z up, metres) and solid boxes standing on or over it."""
    parts = [trimesh.creation.box(extents=[4.0, 4.0, 0.02], transform=trimesh.transformations.translation_matrix(
        [0.0, 0.0, -0.01]))]
    for low, high in boxes:
        centre = [(a + b) / 2 for a, b in zip(low, high)]
        parts.append(trimesh.creation.box(extents=[b - a for a, b in zip(low, high)],
                                          transform=trimesh.transformations.translation_matrix(centre)))
    return trimesh.util.concatenate(parts)


def _start(collider, attractor, region=([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0])):
    return choose_start(collider=collider, region=region, attractor=attractor, design=BALL, up=2, up_sign=1,
                        meters_per_unit=1.0, support_under=support_under)


def test_the_ball_starts_on_the_clear_floor_closest_to_the_destination():
    # A desk top at 0.74 m over x in [0.2, 0.8]; the destination is on it.
    desk = ([0.2, -0.3, 0.70], [0.8, 0.3, 0.74])
    lower, upper, support, report = _start(_floor_with(desk), attractor=[0.5, 0.0, 0.74])
    assert support is not None and report["rule"] == "closest_clear_spot_on_named_support_to_destination"
    assert lower[2] == pytest.approx(0.0, abs=1e-6) and upper[2] == pytest.approx(0.065, abs=1e-6)
    centre = [(a + b) / 2 for a, b in zip(lower, upper)]
    # Never under the desk: the hand needs the column above the ball.
    assert not (0.2 - 0.0325 < centre[0] < 0.8 + 0.0325 and -0.3 - 0.0325 < centre[1] < 0.3 + 0.0325)
    assert np.hypot(centre[0] - 0.5, centre[1]) < 0.45 and report["clear_column_m"] == 1.0


def test_an_object_on_the_floor_keeps_the_ball_away_and_a_full_region_is_refused():
    backpack = ([-0.2, -0.2, 0.0], [0.2, 0.2, 0.3])
    lower, upper, support, _ = _start(_floor_with(backpack), attractor=[0.0, 0.0, 0.0])
    centre = [(a + b) / 2 for a, b in zip(lower, upper)]
    assert support is not None and max(abs(centre[0]), abs(centre[1])) > 0.2 + 0.0325
    # Covered everywhere: no spot, named by the caller.
    lower, upper, support, report = _start(_floor_with(([-1.2, -1.2, 0.0], [1.2, 1.2, 0.1])), attractor=[0, 0, 0])
    assert support is None and lower is None and report["clear_spots_seen"] == 0


def test_construction_authors_the_design_as_supplied_geometry_within_its_bounds(tmp_path):
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
    from blueprint_pipeline.task_evaluation_completed_scene_adapters import execute_provided_mesh_authoring
    from blueprint_pipeline.website_synthetic_subject import AUTHORING_ADAPTER, AUTHORING_METHOD

    physics = design_physics(BALL)
    configuration = records.stage_three_configuration(scene_id="site-1",
        replacement_identity={"id": "website-subject-ball", "version": "v1"}, source_instance_id="synthetic-ball",
        authoring_target="ball", source_min=[0.4, 0.0, 0.0], source_max=[0.465, 0.065, 0.065],
        dimension_tolerance=0.1, physics_bounds={key + "_bounds": value for key, value in physics["bounds"].items()})
    configuration.pop("authoring_backend")
    configuration.update(authoring_method=AUTHORING_METHOD)
    candidate = tmp_path / "source_object_candidate.usda"
    geometry = write_candidate(BALL, center_xyz_m=[0.4325, 0.0325, 0.0325], path=candidate)
    assert geometry["watertight"] is True
    data = candidate.read_bytes()
    dependency = {"output_artifacts": [{"role": "source_object_candidate_mesh", "path": str(candidate),
                                        "digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}]}
    stage = {**records.stage_sequence()[2], "adapter": {"id": AUTHORING_ADAPTER, "version": "v1"},
             "execution_class": "no_spend"}
    configuration_path = tmp_path / "stage-3.json"
    configuration_path.write_text(json.dumps(configuration))
    result = execute_provided_mesh_authoring(envelope={}, stage=stage, configuration=configuration,
        configuration_path=configuration_path, dependency_results=(dependency,), output_root=tmp_path / "out")
    receipt = json.loads((tmp_path / "out" / "replacement_authoring_receipt.v1.json").read_text())
    assert result["status"] == "completed" and receipt["model_called"] is False
    completion = receipt["candidate_physics_completion"]
    assert completion["metric_envelope_validation"]["status"] == "within_preregistered_metric_envelope"
    assert completion["collision_dimensions_m"] == pytest.approx([0.065] * 3, abs=0.002)


def test_a_designed_subject_intent_is_resolved_as_a_website_source(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_source_resolver as resolver
    from blueprint_pipeline import website_scene_dispatch

    seen = []
    monkeypatch.setattr(website_scene_dispatch, "resolve_website_source",
                        lambda **kwargs: seen.append(kwargs) or "website-source")
    intent = {"request": {"source": {"kind": "gaussian_splat", "binding_id": "website-splat-0"},
                          "task": {"subject": {"geometry_origin": "synthetic_design"}}}}
    assert resolver.resolve_scene_source(intent=intent, config={}, release={}) == "website-source" and seen


def test_a_pending_designed_subject_preparation_passes_the_preparation_contract(tmp_path, monkeypatch):
    import tests.test_website_task_preparation as fixture
    from blueprint_pipeline.task_evaluation_scene_intake import validate_request
    from blueprint_pipeline.website_preparation_contracts import validate_preparation_request
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
    from tests.test_website_preparation_contract_v2 import pending_geometry

    real = fixture._arguments
    monkeypatch.setattr(fixture, "_arguments",
                        lambda root, *a, **k: fixture.synthetic_arguments(real(root, *a, **k)))
    args = pending_geometry(tmp_path)
    prepared = compile_website_scene_preparation(**args)
    assert prepared["status"] == "intake_ready", prepared["blockers"]
    request = validate_request(prepared["intake_request"], now=args["now"])
    assert request["task"]["subject"]["geometry_origin"] == "synthetic_design"
    validate_preparation_request(request=request, task_context=args["task_context"],
                                 authority=prepared["website_preparation_authority"], now=args["now"])
