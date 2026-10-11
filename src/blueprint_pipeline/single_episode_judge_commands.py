"""Command edits and native-judge transport checks for the direct episode launcher."""
from __future__ import annotations

import argparse
import shlex
from collections.abc import Mapping
from typing import Any


def _replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            raise ValueError(f"closed_loop_option_value_missing:{option}")
        result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def _replace_repeated_option(command: list[str], option: str, values: list[str]) -> list[str]:
    result = _remove_option(command, option, takes_value=True)
    for value in values:
        result.extend([option, value])
    return result


def _remove_option(command: list[str], option: str, *, takes_value: bool) -> list[str]:
    result = list(command)
    while option in result:
        index = result.index(option)
        del result[index]
        if takes_value:
            if index >= len(result):
                raise ValueError(f"closed_loop_option_value_missing:{option}")
            del result[index]
    return result


def _add_flag(command: list[str], option: str) -> list[str]:
    result = _remove_option(command, option, takes_value=False)
    result.append(option)
    return result


def _direct_native_judge_transport_blockers(plan: Mapping[str, Any]) -> list[str]:
    """The direct Render launcher has no admitted private-file transport yet."""
    from .haiku_vision_judge import MODEL, replacement_model

    command = [str(item) for item in plan.get("closed_loop_command") or []]
    worker_env = plan.get("env") or {}
    command_parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    command_parser.add_argument("--wam-consistency-command")
    command_parser.add_argument("--wam-success-label-command")
    try:
        configured, _ = command_parser.parse_known_args(command)
    except argparse.ArgumentError:
        return ["single_episode_native_judge_command_argument_invalid"]
    for option, module, model_env in (
        ("--wam-consistency-command", "blueprint_pipeline.wam_episode_consistency_label_openai",
         "BLUEPRINT_OPENAI_WAM_EPISODE_CONSISTENCY_MODEL"),
        ("--wam-success-label-command", "blueprint_pipeline.wam_generated_video_success_label_openai",
         "BLUEPRINT_OPENAI_WAM_SUCCESS_LABEL_MODEL"),
    ):
        judge_command = getattr(configured, option[2:].replace("-", "_"))
        if not judge_command:
            continue
        try:
            tokens = shlex.split(judge_command)
        except ValueError:
            return ["single_episode_native_judge_command_argument_invalid"]
        if module not in tokens:
            continue
        model = str(worker_env.get(model_env) or MODEL)
        # Match argparse's equals syntax and last-option-wins semantics.
        parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
        parser.add_argument("--model")
        try:
            parsed, _ = parser.parse_known_args(tokens[tokens.index(module) + 1:])
        except argparse.ArgumentError:
            return ["single_episode_native_judge_model_argument_invalid"]
        model = str(parsed.model or model).strip() or MODEL
        if replacement_model(model) == MODEL:
            return ["single_episode_native_haiku_secret_transport_unqualified"]
    return []


