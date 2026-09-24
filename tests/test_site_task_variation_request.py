"""Variation requests derived from a site task: bounded, supported, admitted."""

from __future__ import annotations

import pytest

from blueprint_pipeline.exact_workcell_variation_matrix import (
    ExactWorkcellVariationError,
    compile_variation_matrix,
)
from blueprint_pipeline.native_task_runtime_contract import SUPPORTED_SCENARIO_RUNTIME_TARGETS
from blueprint_pipeline.site_task_variation_request import build_site_task_variation_request
from tests.test_task_evaluation_policy_canary_rescore import _task_spec

D = "sha256:"


def _bindings() -> dict[str, dict[str, str]]:
    return {
        "scene_binding": {"scene_id": "scene839873", "scene_digest": D + "1" * 64,
                          "coordinate_frame_digest": D + "2" * 64,
                          "canonical_object_asset_id": "cup", "canonical_object_asset_digest": D + "3" * 64},
        "task_binding": {"task_id": "cup-to-shelf", "task_digest": D + "4" * 64,
                         "reset_contract_digest": D + "5" * 64, "success_contract_digest": D + "6" * 64},
        "embodiment_binding": {"embodiment_id": "franka-droid", "embodiment_digest": D + "7" * 64,
                               "joint_limits_digest": D + "8" * 64, "camera_calibration_digest": D + "9" * 64},
    }


def _build(**overrides):
    arguments = {"matrix_id": "scene839873-cup-v1", "implementation_commit": "a" * 40, "seed_root": 7,
                 "task_spec": _task_spec(), **_bindings()}
    arguments.update(overrides)
    return build_site_task_variation_request(**arguments)


def test_a_site_task_yields_an_admitted_request_on_supported_targets_only() -> None:
    request, omitted = _build(nominal_dynamic_friction=0.5)
    targets = [row["application_target"] for row in request["variation_dimensions"]]
    assert set(targets) <= SUPPORTED_SCENARIO_RUNTIME_TARGETS
    assert [row["family"] for row in request["variation_dimensions"]] == [
        "placement_approach", "placement_approach", "illumination", "camera_sensor", "bounded_physics",
    ]
    assert omitted == []
    assert all(row["changes_object_or_task_identity"] is False for row in request["variation_dimensions"])
    matrix = compile_variation_matrix(request)
    assert len(matrix["cells"]) == 100


def test_bounds_come_from_the_task() -> None:
    request, _ = _build()
    by_id = {row["dimension_id"]: row for row in request["variation_dimensions"]}
    # The fixture's destination is 0.15 m away, so the start moves at most a quarter of that.
    assert by_id["object_start_y_delta"]["maximum"] == pytest.approx(0.03)
    near, _ = _build(task_spec={**_task_spec(), "destination_position_bounds_world_m": {
        "minimum": [1.03, 1.99, 0.79], "maximum": [1.05, 2.01, 0.81]}})
    assert {row["dimension_id"]: row for row in near["variation_dimensions"]}[
        "object_start_y_delta"]["maximum"] == pytest.approx(0.01)
    # 0.1 rad orientation tolerance keeps the start rotation within half of it.
    assert by_id["object_yaw_delta"]["maximum"] == pytest.approx(2.86, abs=0.01)


def test_unknowns_are_left_out_rather_than_filled_in() -> None:
    request, omitted = _build()
    assert "dynamic_friction:no_measured_nominal" in omitted
    assert all(row["family"] != "bounded_physics" for row in request["variation_dimensions"])
    touching, omitted = _build(task_spec={**_task_spec(), "destination_position_bounds_world_m": {
        "minimum": [1.0, 2.0, 0.8], "maximum": [1.0, 2.0, 0.8]}}, nominal_dynamic_friction=0.5)
    assert "object_start_y:destination_too_close_to_vary_start" in omitted
    tight, omitted = _build(task_spec={**_task_spec(), "destination_orientation_tolerance_rad": 0.02})
    assert "object_yaw:orientation_tolerance_too_tight_to_vary" in omitted


def test_the_matrix_contract_still_owns_admission() -> None:
    bindings = _bindings()
    bindings["task_binding"] = {**bindings["task_binding"], "task_digest": "not-a-digest"}
    with pytest.raises(ExactWorkcellVariationError, match="task_task_digest"):
        _build(**bindings)
    with pytest.raises(ExactWorkcellVariationError, match="request_cell_count_invalid"):
        _build(cell_count=5)
