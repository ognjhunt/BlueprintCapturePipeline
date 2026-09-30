"""Preparation record identity lookup without preparation execution imports."""
from __future__ import annotations

from . import task_evaluation_scene_intent_contracts as intake
from .task_evaluation_scene_progression_contracts import require, safe_path

SCHEMA = "task_evaluation_scene_preparation_attempt.v1"


def preparation_attempt_path(directory, attempt_id):
    require(intake._identifier(attempt_id), "preparation_attempt_id_invalid")
    directory = safe_path(directory)
    paths = [directory / name / (attempt_id + ".json") for name in ("preparation-attempts", "attempts")]
    found = [path for path in paths if path.exists()]
    require(len(found) <= 1, "preparation_attempt_identity_ambiguous")
    return found[0] if found else paths[0]
