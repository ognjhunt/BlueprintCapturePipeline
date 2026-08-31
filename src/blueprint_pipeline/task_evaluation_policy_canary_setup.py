"""Validate the secret-clean policy canary setup projected in a launch profile."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_run_contract import QUICK_FAMILY_COUNTS


SCHEMA_VERSION = "task_evaluation_policy_canary_setup.v1"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schemas" / f"{SCHEMA_VERSION}.schema.json"


class TaskEvaluationPolicyCanarySetupError(ValueError):
    pass


@lru_cache(maxsize=1)
def policy_canary_setup_schema() -> dict[str, Any]:
    import jsonschema

    try:
        value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(value)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        raise TaskEvaluationPolicyCanarySetupError(
            "policy_canary_setup_schema_invalid"
        ) from exc
    return deepcopy(value)


def validate_policy_canary_setup(value: Mapping[str, Any]) -> dict[str, Any]:
    import jsonschema

    setup = deepcopy(dict(value))
    errors = sorted(
        jsonschema.Draft202012Validator(policy_canary_setup_schema()).iter_errors(setup),
        key=lambda row: list(row.path),
    )
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "$"
        raise TaskEvaluationPolicyCanarySetupError(
            f"policy_canary_setup_invalid:{path}"
        )
    if setup["setup_digest"] != canonical_digest(setup, digest_field="setup_digest"):
        raise TaskEvaluationPolicyCanarySetupError("policy_canary_setup_digest_mismatch")
    presets = setup["episode_presets"]
    if [row["preset_id"] for row in presets] != ["quick_10", "standard_100", "deep_500"]:
        raise TaskEvaluationPolicyCanarySetupError("policy_canary_setup_preset_order_invalid")
    quick = presets[0]
    matrix = quick["matrix"]
    cells = matrix["cells"]
    if (
        quick["availability"] != "enabled"
        or quick["recommended"] is not True
        or len(cells) != 10
        or dict(Counter(row["family"] for row in cells)) != QUICK_FAMILY_COUNTS
        or matrix["expected_family_counts"] != QUICK_FAMILY_COUNTS
        or matrix["matrix_digest"] != canonical_digest({"ordered_cells": cells})
        or len({row["cell_id"] for row in cells}) != 10
    ):
        raise TaskEvaluationPolicyCanarySetupError("policy_canary_setup_quick_matrix_invalid")
    if any(
        row["availability"] != "coming_later"
        or row["recommended"] is not False
        or row["matrix"]["cells"] != []
        for row in presets[1:]
    ):
        raise TaskEvaluationPolicyCanarySetupError("policy_canary_setup_large_preset_enabled")
    runnable: list[str] = []
    for robot in setup["robot_presets"]:
        if robot["readiness"]["status"] != "verified_runnable":
            continue
        for policy in robot["policy_candidates"]:
            if policy["readiness"]["status"] != "verified_runnable":
                continue
            compatibility = policy["compatibility"]
            expected = {
                "robot_preset_ids": robot["robot_preset_id"],
                "embodiment_ids": robot["embodiment_id"],
                "observation_schema_ids": robot["observation_schema"]["schema_id"],
                "action_schema_ids": robot["action_schema"]["schema_id"],
                "simulator_runtime_ids": robot["simulator_runtime_id"],
                "task_family_ids": robot["task_family_id"],
            }
            if any(value not in compatibility[field] for field, value in expected.items()):
                raise TaskEvaluationPolicyCanarySetupError(
                    "policy_canary_setup_compatibility_invalid"
                )
            runnable.append(policy["candidate_id"])
    if runnable != ["pi05_droid", "groot_n17_droid"]:
        raise TaskEvaluationPolicyCanarySetupError("policy_canary_setup_runnable_pair_invalid")
    return setup


def policy_canary_setup_blockers(
    value: Any, *, prefix: str, source_launch_id: str | None = None
) -> list[str]:
    try:
        setup = validate_policy_canary_setup(value if isinstance(value, Mapping) else {})
    except TaskEvaluationPolicyCanarySetupError as exc:
        return [f"{prefix}:{exc}"]
    if source_launch_id is not None and setup["source_launch_id"] != source_launch_id:
        return [f"{prefix}:source_launch_id_mismatch"]
    return []


def launch_profile_policy_canary_setup_blockers(
    profile: Mapping[str, Any], *, prefix: str
) -> list[str]:
    if "internal_policy_canary_setup" not in profile:
        return []
    return policy_canary_setup_blockers(
        profile["internal_policy_canary_setup"],
        prefix=prefix,
        source_launch_id=str(
            profile.get("configured_source_launch_id")
            or profile.get("profile_id")
            or ""
        ),
    )


__all__ = ["SCHEMA_PATH", "SCHEMA_VERSION", "TaskEvaluationPolicyCanarySetupError", "launch_profile_policy_canary_setup_blockers", "policy_canary_setup_blockers", "policy_canary_setup_schema", "validate_policy_canary_setup"]
