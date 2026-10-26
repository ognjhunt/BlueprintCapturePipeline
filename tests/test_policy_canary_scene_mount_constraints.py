"""Scene mounting constrains diagnostic variation before paid execution."""

from copy import deepcopy

import pytest

from blueprint_pipeline.policy_canary_scene_resolution import resolved_scene_plan
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _quick_cells


REVISION = "sha256:" + "3" * 64


def mounted_plan():
    return {
        "task_kind": "articulated_open_close",
        "task_spec": {
            "task_kind": "articulated_open_close",
            "manipulation_strategy": "articulated_open_close",
            "instruction_subject_label": "installed fixture",
            "visible_target_label": "panel",
            "configured_success_criteria": {"joint_type": "revolute", "closing_required": False},
        },
        "robot": {"base_pose_world": {
            "position_world_m": [0., 0., 0.], "orientation_xyzw": [0., 0., 0., 1.]}},
        "cameras": [],
        "cadence": {
            "control_frequency_hz": 15., "physics_frequency_hz": 120.,
            "physics_dt_seconds": 1 / 120, "maximum_action_steps": 30,
            "settle_window_samples": 4,
        },
        "objects": [{
            "name": "arbitrary_fixture", "task_subject": True,
            "object_type": "ARTICULATION",
            "articulation_adaptation": {
                "adaptation": "usd_authored_world_fixed_base",
                "scene_mounting": {"mode": "fixed_in_scene"},
            },
            "pose_world": {"position_world_m": [1., 2., 3.],
                           "orientation_xyzw": [0., 0., 0., 1.]},
            "reset_state": {"root_pose_world": {
                "position_world_m": [1., 2., 3.],
                "orientation_xyzw": [0., 0., 0., 1.]}},
        }],
    }


def test_diagnostic_cells_keep_installed_root_and_report_unexercised_variations():
    plan = mounted_plan()
    original = deepcopy(plan)
    cells = _quick_cells(REVISION, scene_id="different-site", scene_plan=plan)
    assert len(cells) == 10
    assert plan == original
    assert cells == _quick_cells(REVISION, scene_id="different-site", scene_plan=plan)
    for cell in cells:
        scenario = cell["resolved_scenario"]
        assert not any(key.startswith("object_start_") or key == "object_yaw_delta_degrees"
                       for key in scenario["parameters"])
    removed = cells[2]["resolved_scenario"]["runtime_coverage_gaps"]
    assert removed[0]["parameters_not_exercised"] == {"object_start_y_delta_m": -.02}
    assert removed[0]["qualified_variation_claim_permitted"] is False
    assert cells[8]["resolved_scenario"]["parameters"] == {"task_light_intensity_scale": 1.1}
    resolved = resolved_scene_plan(plan, cells[2], task_success_contract={
        "contract_digest": "sha256:" + "7" * 64})
    assert resolved["objects"] == plan["objects"]
    assert resolved["scenario"]["runtime_coverage_gaps"] == removed


def test_loose_object_keeps_the_existing_cell_bytes_and_seeds():
    plan = mounted_plan()
    del plan["objects"][0]["articulation_adaptation"]
    assert _quick_cells(REVISION, scene_plan=plan) == _quick_cells(REVISION)


@pytest.mark.parametrize("mounting_mode", ["free", "unknown"])
def test_simulation_anchor_does_not_invent_a_site_installation(mounting_mode):
    plan = mounted_plan()
    plan["objects"][0]["articulation_adaptation"]["scene_mounting"]["mode"] = mounting_mode
    assert _quick_cells(REVISION, scene_plan=plan) == _quick_cells(REVISION)


@pytest.mark.parametrize("parameters", [
    {"object_start_x_delta_m": .02},
    {"object_start_y_delta_m": -.02},
    {"object_yaw_delta_degrees": 7.5},
])
def test_resolver_refuses_unadapted_fixed_root_variations_before_camera_or_runtime(parameters):
    plan = mounted_plan()
    original = deepcopy(plan)
    contract = {"contract_digest": "sha256:" + "7" * 64}
    cell = {"cell_id": "foreign-plan", "seed": 7,
            "resolved_scenario": {"family": "placement_approach", "parameters": parameters}}
    with pytest.raises(RuntimeError, match="policy_canary_fixed_subject_root_variation_forbidden"):
        resolved_scene_plan(plan, cell, task_success_contract=contract)
    assert plan == original


def test_fixed_root_shift_is_rejected_before_isaac_or_policy_queries(tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import (
        FakeIsaac, _rehearsal_runtime, _stage_runtime_root,
    )

    runtime_root, provider_output = _stage_runtime_root(tmp_path)
    child = provider_output / "cell_runs" / "00"
    child.mkdir(parents=True)
    isaac = FakeIsaac(child / worker.PROVIDER_RESULT_FILENAME)
    resolve = worker._resolved_scene_plan

    def supplied_fixture_shift(base, cell, **kwargs):
        base, cell = deepcopy(base), deepcopy(cell)
        subject = next(row for row in base["objects"] if row.get("task_subject") is True)
        subject["articulation_adaptation"] = {"scene_mounting": {"mode": "fixed_in_scene"}}
        cell["resolved_scenario"]["parameters"]["object_start_y_delta_m"] = .02
        return resolve(base, cell, **kwargs)

    monkeypatch.setattr(worker, "_resolved_scene_plan", supplied_fixture_shift)
    with pytest.raises(RuntimeError, match="policy_canary_fixed_subject_root_variation_forbidden"):
        worker._run_selected_cell(0, runtime_root=runtime_root, output_root=child,
                                 provider_output_root=provider_output,
                                 cell_runtime=_rehearsal_runtime(isaac))
    assert isaac.launches == 0
