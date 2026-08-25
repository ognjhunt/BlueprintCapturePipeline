from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_construction_recipe import (
    CAPABILITY_ORDER,
    SCHEMA_VERSION,
    TaskEvaluationSceneConstructionRecipeError,
    validate_scene_construction_recipe,
)


def _ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/construction-{index}.json",
        "digest": f"sha256:{index:064x}",
        "size_bytes": 1000 + index,
    }


def _recipe() -> dict[str, object]:
    stages = []
    for index, capability in enumerate(CAPABILITY_ORDER):
        stage_id = f"stage-{index + 1}"
        stages.append(
            {
                "stage_id": stage_id,
                "capability": capability,
                "adapter": {"id": f"adapter-{index + 1}", "version": "v1"},
                "execution_class": "gpu_canary" if index in {0, 2, 4} else "no_spend",
                "configuration": _ref(index + 1),
                "depends_on": [] if index == 0 else [f"stage-{index}"],
            }
        )
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "recipe_id": "scene-source-object-replacement-v1",
        "team_namespace": "team-a",
        "scene_identity": {"id": "scene-a", "version": "v1"},
        "task_identity": {"id": "task-a", "version": "v1"},
        "subject_identity": {"id": "source-object-replacement", "version": "v1"},
        "source_manifest_digest": "sha256:" + "a" * 64,
        "rights_admission_digest": "sha256:" + "b" * 64,
        "stage_sequence": stages,
        "output_identity": {"id": "native-arena-packet", "version": "v1"},
        "provider_disclosure": {
            "raw_source_bytes_to_external_provider": False,
            "derived_runtime_processing_allowed": True,
            "provider_training_allowed": False,
            "public_redistribution_allowed": False,
        },
        "recipe_digest": "",
    }
    value["recipe_digest"] = canonical_digest(value, digest_field="recipe_digest")
    return value


def test_accepts_provider_neutral_production_construction_chain() -> None:
    value = _recipe()
    assert validate_scene_construction_recipe(value) == value


@pytest.mark.parametrize(
    ("mutate", "blocker"),
    [
        (
            lambda value: value["stage_sequence"].reverse(),
            "scene_construction_recipe_capability_order_invalid",
        ),
        (
            lambda value: value["stage_sequence"][3].update(depends_on=[]),
            "scene_construction_recipe_dependency_invalid:stage-4",
        ),
        (
            lambda value: value["provider_disclosure"].update(
                raw_source_bytes_to_external_provider=True
            ),
            "scene_construction_recipe_invalid:provider_disclosure.raw_source_bytes_to_external_provider",
        ),
    ],
)
def test_recipe_fails_closed_before_production_execution(mutate, blocker) -> None:
    value = copy.deepcopy(_recipe())
    mutate(value)
    value["recipe_digest"] = canonical_digest(value, digest_field="recipe_digest")
    with pytest.raises(TaskEvaluationSceneConstructionRecipeError, match=blocker):
        validate_scene_construction_recipe(value)


def test_recipe_digest_binds_every_adapter_and_configuration() -> None:
    value = _recipe()
    value["stage_sequence"][2]["adapter"]["version"] = "v2"
    with pytest.raises(
        TaskEvaluationSceneConstructionRecipeError,
        match="scene_construction_recipe_digest_invalid",
    ):
        validate_scene_construction_recipe(value)
