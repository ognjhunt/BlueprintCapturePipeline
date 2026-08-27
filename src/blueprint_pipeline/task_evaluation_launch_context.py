"""Fail-closed public identity context for configured Task Evaluation Runs."""

from __future__ import annotations

from typing import Any, Mapping


TASK_EVALUATION_RUN_CONTEXT_FIELDS = frozenset(
    {
        "run_mode",
        "team_namespace",
        "scene_id",
        "task_id",
        "configuration_run_id",
        "evaluation_episode_executed",
    }
)


def is_identifier(value: Any) -> bool:
    text = str(value or "")
    return (
        bool(text)
        and len(text) <= 192
        and all(character.isalnum() or character in "._-" for character in text)
    )


def validate_task_evaluation_run_context(
    value: Any, *, blocker_prefix: str
) -> list[str]:
    blockers: list[str] = []
    context = dict(value) if isinstance(value, Mapping) else {}
    if not isinstance(value, Mapping) or set(context) != TASK_EVALUATION_RUN_CONTEXT_FIELDS:
        return [f"{blocker_prefix}_fields_invalid"]
    if context.get("run_mode") != "scene_configuration":
        blockers.append(f"{blocker_prefix}_run_mode_invalid")
    for field in (
        "team_namespace",
        "scene_id",
        "task_id",
        "configuration_run_id",
    ):
        if not is_identifier(context.get(field)):
            blockers.append(f"{blocker_prefix}_{field}_invalid")
    if context.get("evaluation_episode_executed") is not False:
        blockers.append(f"{blocker_prefix}_evaluation_episode_executed_invalid")
    return blockers


__all__ = ["is_identifier", "validate_task_evaluation_run_context"]
