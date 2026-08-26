"""Validate production-owned scene construction recipes for Task Evaluation.

The authenticated website request supplies admitted source references and this
immutable recipe.  Production executes the recipe after submission; callers do
not upload a prebuilt native-Arena packet.  Capabilities are stable while each
adapter remains replaceable and versioned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_scene_construction_recipe.v1"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "schemas"
    / "task_evaluation_scene_construction_recipe.v1.schema.json"
)
CAPABILITY_ORDER = (
    "observed_appearance_object_removal",
    "collision_object_excision",
    "rigid_replacement_authoring",
    "replacement_static_qualification",
    "replacement_native_import_qualification",
    "scene_assembly",
)


class TaskEvaluationSceneConstructionRecipeError(ValueError):
    """The recipe cannot safely drive production construction."""


@lru_cache(maxsize=1)
def construction_recipe_schema() -> dict[str, Any]:
    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_schema_unavailable"
        ) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_schema_invalid"
        )
    jsonschema.Draft202012Validator.check_schema(value)
    return dict(value)


def validate_scene_construction_recipe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a closed recipe and its exact dependency chain."""

    try:
        recipe = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_not_json"
        ) from exc
    validator = jsonschema.Draft202012Validator(
        construction_recipe_schema(), format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(recipe), key=lambda row: list(row.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationSceneConstructionRecipeError(
            f"scene_construction_recipe_invalid:{path}"
        )
    stages = recipe["stage_sequence"]
    if tuple(stage["capability"] for stage in stages) != CAPABILITY_ORDER:
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_capability_order_invalid"
        )
    stage_ids = [stage["stage_id"] for stage in stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_stage_id_duplicate"
        )
    for index, stage in enumerate(stages):
        expected = [] if index == 0 else [stage_ids[index - 1]]
        if stage["depends_on"] != expected:
            raise TaskEvaluationSceneConstructionRecipeError(
                f"scene_construction_recipe_dependency_invalid:{stage['stage_id']}"
            )
    if recipe["recipe_digest"] != canonical_digest(
        recipe, digest_field="recipe_digest"
    ):
        raise TaskEvaluationSceneConstructionRecipeError(
            "scene_construction_recipe_digest_invalid"
        )
    return recipe


def scene_construction_recipe_digest(value: Mapping[str, Any]) -> str:
    return validate_scene_construction_recipe(value)["recipe_digest"]


__all__ = [
    "CAPABILITY_ORDER",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConstructionRecipeError",
    "construction_recipe_schema",
    "scene_construction_recipe_digest",
    "validate_scene_construction_recipe",
]
