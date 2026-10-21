"""CPU integration of the frozen matrix, native setters and native readback.

Native buffers are test doubles. These checks prove adapter/recovery contracts,
not simulator execution, policy outcomes or physical fidelity.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_native_runtime import (
    build_native_site_task_variation_request,
    native_cell_parameter_bindings,
    native_runtime_embodiment_digest,
    prepare_native_runtime_cell,
    read_native_runtime_cell,
)
from blueprint_pipeline.exact_workcell_variation_matrix import (
    ExactWorkcellVariationError,
    compile_variation_matrix,
)
from tests.test_exact_workcell_variation_matrix import _dimension, _request


def _fixture(*, site="site-a", y=0.4):
    pose = {"position_world_m": [0.3, y, 0.8], "orientation_xyzw": [0., 0., 0., 1.]}
    plan = {
        "schema_version": "native_task_arena_scene_plan.v1", "scene_id": site,
        "task_id": site + ".move", "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "franka_panda", "control_hz": 15},
        "cadence": {"control_frequency_hz": 15, "physics_frequency_hz": 120},
        "objects": [{"name": "subject", "task_subject": True, "asset_id": "part",
                     "sha256": "sha256:" + "3" * 64, "pose_world": copy.deepcopy(pose),
                     "reset_state": {"root_pose_world": copy.deepcopy(pose)}}],
        "cameras": [{"role": role, "frame_from_camera_matrix": [
            1., 0., 0., 1.2, 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
        ]} for role in ("external", "wrist")],
        "task_spec": {"start_pose_world": [*pose["position_world_m"], *pose["orientation_xyzw"]],
                      "destination_position_bounds_world_m": {"minimum": [.5, y - .01, .8], "maximum": [.6, y + .01, .8]},
                      "destination_orientation_tolerance_rad": .2,
                      "task_success_contract": {"contract_digest": "sha256:" + "6" * 64}},
        "scenario": {"context_kind": "construction_canary", "parameter_applications": []},
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    request = _request(task_id=plan["task_id"], embodiment_id="franka_panda")
    request["matrix_id"] = site + ".move.baseline_100"
    request["scene_binding"].update(scene_id=site, canonical_object_asset_id="part")
    request["task_binding"]["task_digest"] = canonical_digest(plan["task_spec"])
    request["embodiment_binding"]["embodiment_digest"] = native_runtime_embodiment_digest(plan)
    request["variation_dimensions"] = []
    for name, family, nominal, low, high, unit, target in [
        ("start_y", "placement_approach", y, y - .02, y + .02, "m", "object_start_position_m.y"),
        ("yaw", "placement_approach", 0., -5., 5., "degrees", "object_orientation.yaw"),
        ("light", "illumination", 1., .8, 1.2, "ratio", "task_light.intensity_scale"),
        ("camera_x", "camera_sensor", 1.2, 1.18, 1.22, "m", "external_camera.pose.position.x"),
    ]:
        row = _dimension(name, family, nominal=nominal, minimum=low, maximum=high, unit=unit)
        row["application_target"] = "EventManager.reset." + target
        request["variation_dimensions"].append(row)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request, compile_variation_matrix(request), plan


def _native_buffers(plan):
    pose = plan["objects"][0]["pose_world"]
    position, q = pose["position_world_m"], pose["orientation_xyzw"]
    # The pinned Isaac Lab Beta2 adapter reads native XYZW storage.
    native_pose = np.array([[*position, *q]], dtype=np.float64)
    subject = SimpleNamespace(data=SimpleNamespace(root_pose_w=native_pose))
    return SimpleNamespace(
        plan=plan, env=SimpleNamespace(scene={"subject": subject}), scene_asset_names={"subject": "subject"},
        native_configuration_readback={
            "cameras": {"external": {"offset_position_m": [plan["cameras"][0]["frame_from_camera_matrix"][3], 0., 0.]}},
            "scenario_parameters": {"light": {"observed_intensity_scale": next(
                row["resolved_value"] for row in plan["scenario"]["parameter_bindings"] if row["parameter_id"] == "light"
            )}},
        },
    )


@pytest.mark.parametrize("site,y", [("site-a", .4), ("site-b", 1.7)])
def test_two_sites_use_the_same_native_adapter_and_exact_reset_receipt(site, y):
    request, matrix, source = _fixture(site=site, y=y)
    immutable = copy.deepcopy(source)
    # A non-anchor composition actually changes more than one native setter.
    cell = matrix["cells"][-1]
    plan = prepare_native_runtime_cell(
        matrix=matrix, request=request, cell_id=cell["cell_id"],
        source_scene_plan=source, source_plan_digest=source["plan_digest"],
    )
    assert source == immutable
    assert plan["objects"][0]["pose_world"]["position_world_m"][1] == cell["resolved_values"]["start_y"]
    assert plan["task_spec"]["start_pose_world"][1] == cell["resolved_values"]["start_y"]
    result = read_native_runtime_cell(
        built=_native_buffers(plan), matrix=matrix, request=request, cell_id=cell["cell_id"],
        prepared_plan_digest=plan["plan_digest"],
    )
    assert result["status"] == "applied_and_readback_verified"
    assert result["cell_digest"] == cell["cell_digest"]
    assert result["reset_digest"] == cell["reset_digest"]
    assert result["policy_query_performed"] is False
    assert result["application_digest"] == canonical_digest(result, digest_field="application_digest")
    assert all(row["observed"]["source"] == "native_readback" for row in result["observations"])


def test_failed_native_application_cannot_echo_requested_values_as_observed():
    request, matrix, source = _fixture()
    cell = next(row for row in matrix["cells"] if row["resolved_values"]["start_y"] != .4)
    plan = prepare_native_runtime_cell(matrix=matrix, request=request, cell_id=cell["cell_id"],
                                       source_scene_plan=source, source_plan_digest=source["plan_digest"])
    built = _native_buffers(plan)
    built.env.scene["subject"].data.root_pose_w[0, 1] = .4  # Native reset did not apply the varied start.
    result = read_native_runtime_cell(built=built, matrix=matrix, request=request, cell_id=cell["cell_id"],
                                     prepared_plan_digest=plan["plan_digest"])
    assert result["status"] == "blocked"
    assert "runtime_cell_native_value_mismatch:apply_start_y" in result["blockers"]


def test_unsupported_target_never_mutates_the_source_plan():
    request, _, source = _fixture()
    request["variation_dimensions"][0]["application_target"] = "EventManager.reset.hinge_angle"
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    immutable = copy.deepcopy(source)
    with pytest.raises(ExactWorkcellVariationError, match="scenario_target_unsupported"):
        prepare_native_runtime_cell(matrix=matrix, request=request, cell_id=matrix["cells"][0]["cell_id"],
                                    source_scene_plan=source, source_plan_digest=source["plan_digest"])
    assert source == immutable


@pytest.mark.parametrize("mutate", [
    lambda request: request["scene_binding"].update(canonical_object_asset_digest="sha256:" + "e" * 64),
    lambda request: request["task_binding"].update(task_digest="sha256:" + "e" * 64),
    lambda request: request["embodiment_binding"].update(embodiment_digest="sha256:" + "e" * 64),
])
def test_task_object_and_embodiment_changes_refuse_an_old_source_plan(mutate):
    request, _, source = _fixture()
    mutate(request)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    with pytest.raises(ExactWorkcellVariationError, match="native_exact_workcell_.*mismatch"):
        prepare_native_runtime_cell(matrix=matrix, request=request, cell_id=matrix["cells"][0]["cell_id"],
                                    source_scene_plan=source, source_plan_digest=source["plan_digest"])


def test_mutated_plan_and_duplicate_targets_cannot_reuse_a_reset_binding():
    request, matrix, source = _fixture()
    cell = matrix["cells"][0]
    plan = prepare_native_runtime_cell(matrix=matrix, request=request, cell_id=cell["cell_id"],
                                       source_scene_plan=source, source_plan_digest=source["plan_digest"])
    prepared = plan["plan_digest"]
    plan["scenario"]["seed"] += 1
    with pytest.raises(ExactWorkcellVariationError, match="runtime_binding_mismatch"):
        read_native_runtime_cell(built=_native_buffers(plan), matrix=matrix, request=request,
                                 cell_id=cell["cell_id"], prepared_plan_digest=prepared)
    request["variation_dimensions"][1].update(
        application_target="EventManager.reset.object_start_position_m.y", unit="m",
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    with pytest.raises(ExactWorkcellVariationError, match="target_ambiguous"):
        native_cell_parameter_bindings(matrix=matrix, request=request, cell_id=matrix["cells"][0]["cell_id"])


@pytest.mark.parametrize("target,unit,code", [
    ("EventManager.reset.object_start_position_m.y", "rad", "scenario_parameter_invalid"),
    ("EventManager.reset.task_subject_link_material.dynamic_friction", "coefficient", "link_selector_required"),
])
def test_native_adapter_does_not_guess_units_or_articulated_link(target, unit, code):
    request, _, _ = _fixture()
    request["variation_dimensions"][0].update(application_target=target, unit=unit)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    matrix = compile_variation_matrix(request)
    with pytest.raises(ExactWorkcellVariationError, match=code):
        native_cell_parameter_bindings(matrix=matrix, request=request, cell_id=matrix["cells"][0]["cell_id"])


@pytest.mark.parametrize("site,y", [("site-a", .4), ("site-b", 1.7)])
def test_source_aware_generator_binds_absolute_native_values_for_another_site(site, y):
    original, _, source = _fixture(site=site, y=y)
    request, omitted = build_native_site_task_variation_request(
        source_scene_plan=source, source_plan_digest=source["plan_digest"], matrix_id=site + ".generated",
        implementation_commit="a" * 40, seed_root=17, scene_binding=original["scene_binding"],
        task_binding=original["task_binding"], embodiment_binding=original["embodiment_binding"],
    )
    dimensions = {row["application_target"]: row for row in request["variation_dimensions"]}
    start = dimensions["EventManager.reset.object_start_position_m.y"]
    camera = dimensions["EventManager.reset.external_camera.pose.position.x"]
    assert start["nominal"] == y
    assert start["minimum"] == pytest.approx(y - .03)
    assert start["maximum"] == pytest.approx(y + .03)
    assert camera["nominal"] == 1.2
    assert camera["minimum"] == 1.18
    assert camera["maximum"] == 1.22
    assert "dynamic_friction:no_measured_nominal" in omitted
    matrix = compile_variation_matrix(request)
    for cell in (matrix["cells"][0], matrix["cells"][1], matrix["cells"][-1]):
        plan = prepare_native_runtime_cell(matrix=matrix, request=request, cell_id=cell["cell_id"],
                                           source_scene_plan=source, source_plan_digest=source["plan_digest"])
        assert plan["objects"][0]["pose_world"]["position_world_m"][1] == pytest.approx(cell["resolved_values"]["object_start_y_delta"])
        assert plan["scenario"]["seed"] == cell["seed"]


def test_source_aware_generator_does_not_guess_a_missing_pose_or_use_stale_calibration():
    original, _, source = _fixture()
    arguments = dict(matrix_id="site-a.generated", implementation_commit="a" * 40, seed_root=17,
                     scene_binding=original["scene_binding"], task_binding=original["task_binding"],
                     embodiment_binding=original["embodiment_binding"])
    source["cameras"][0]["frame_from_camera_matrix"][3] += .1
    with pytest.raises(ExactWorkcellVariationError, match="source_plan_digest_mismatch"):
        build_native_site_task_variation_request(source_scene_plan=source, source_plan_digest=source["plan_digest"], **arguments)
    source["plan_digest"] = canonical_digest(source, digest_field="plan_digest")
    with pytest.raises(ExactWorkcellVariationError, match="embodiment_mismatch"):
        build_native_site_task_variation_request(source_scene_plan=source, source_plan_digest=source["plan_digest"], **arguments)
