"""The articulated replacement verifier, split out of the builtin adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_builtin_adapters as adapters
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_articulated_replacement_result import (
    ACCEPTED_SOURCE_CANDIDATE_CLAIMS,
    verify_articulated_replacement_result,
)


def _verify(**overrides) -> None:
    arguments = {
        "configuration": {},
        "envelope": {"recipe": {"subject_identity": {"id": "cabinet", "version": "v1"}}},
        "receipt": {},
        "graph": {},
        "asset": Path("assembly.usdz"),
        "asset_record": {},
        "source_candidate_record": {},
    }
    arguments.update(overrides)
    verify_articulated_replacement_result(**arguments)


def test_the_builtin_adapters_use_this_verifier_and_claim_vocabulary() -> None:
    assert adapters._verify_articulated_replacement_result is verify_articulated_replacement_result
    assert adapters.ACCEPTED_SOURCE_CANDIDATE_CLAIMS is ACCEPTED_SOURCE_CANDIDATE_CLAIMS


def test_a_development_hypothesis_the_plan_does_not_carry_is_refused() -> None:
    with pytest.raises(TaskEvaluationSceneConfigurationAdapterError,
                       match="development_hypothesis_binding_invalid"):
        _verify(configuration={"development_geometry_hypothesis": {"depth_m": 0.4}})


def test_an_unsealed_result_is_refused() -> None:
    with pytest.raises(TaskEvaluationSceneConfigurationAdapterError,
                       match="content_agents_replacement_result_invalid"):
        _verify()


def test_hinged_appliance_bounds_follow_each_planned_part_role() -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_articulated_replacement_result import (
        expected_articulated_physics_bounds,
    )
    from blueprint_pipeline.task_object_articulated_packaging import plan_articulated_assembly
    from tests.test_articulated_hinged_door_appliance import dishwasher

    configuration = dishwasher()
    plan = plan_articulated_assembly(configuration)
    bounds = expected_articulated_physics_bounds(configuration, plan)
    assert set(bounds) == {"body", "door", "upper_rack", "lower_rack", "cutlery_basket"}
    assert bounds["body"]["mass_kg"] == [10.0, 60.0] and bounds["door"]["mass_kg"] == [2.0, 12.0]
    assert bounds["cutlery_basket"]["mass_kg"] == [0.3, 4.0]
    assert expected_articulated_physics_bounds(configuration, {**plan, "family": "stacked_drawer_cabinet"}) == {
        "assembly_plan_family_mismatch": {}}
    legacy = {key: value for key, value in configuration.items() if key != "assembly_family"}
    assert set(expected_articulated_physics_bounds(legacy, plan)) == {"carcass", "drawer"}
