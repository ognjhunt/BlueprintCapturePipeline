"""OpenAI stage identities and the dedicated managed-asset project boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .task_evaluation_scene_configuration_provider_artifacts import (
    TaskEvaluationSceneConfigurationVastError,
)


# The official-cost gate charges each stage's exclusive (project, key) scope.
OPENAI_RUNTIME_FILE_ENVS = (
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE",
)
OPENAI_RUNTIME_VALUE_ENVS = (
    "OPENAI_PROJECT_ID",
    "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
    "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID",
)
OPENAI_STAGE_SCOPE_DISTINCT_GROUPS = (
    ("OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
     "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE", "OPENAI_CONTENT_AGENTS_API_KEY_FILE"),
    ("BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE",
     "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
     "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE"),
    ("OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
     "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "OPENAI_CONTENT_AGENTS_API_KEY_ID"),
)
OPENAI_STAGE_SCOPE_BINDINGS = (
    ("artifixer_semantic_teacher", "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
     "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE"),
    ("artifixer_visual_review", "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
     "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE"),
    ("content_agents", "OPENAI_CONTENT_AGENTS_API_KEY_ID",
     "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE"),
)

MANAGED_GUARD_FILE_ENV = "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_PROJECT_GUARD_FILE"
_MANAGED_SCOPED_ENVS = {
    "OPENAI_PROJECT_ID": "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_PROJECT_ID",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_ID": "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_KEY_ID",
    "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE": (
        "BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_COST_SCOPE_ATTESTATION_FILE"),
}


def managed_asset_scope(receipt: Mapping[str, Any] | None,
                        stage_caps: Mapping[str, Any]) -> dict[str, str] | None:
    """Select a dedicated project/key only for the signed managed asset route."""
    if (receipt or {}).get("replacement_authoring_agent_runtime") != "openai_agents_api":
        return None
    if ((receipt or {}).get("replacement_authoring_model") != "gpt-6-sol"
            or (receipt or {}).get("replacement_authoring_model_provider", "openai") != "openai"):
        raise TaskEvaluationSceneConfigurationVastError("scene_configuration_agents_api_selection_invalid")
    if any(float(stage_caps[stage]) != 0 for stage in (
            "artifixer_semantic_teacher", "artifixer_visual_review")):
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_agents_api_requires_content_only_scope")
    return {name: str(os.environ.get(alias) or "").strip()
            for name, alias in {**_MANAGED_SCOPED_ENVS, MANAGED_GUARD_FILE_ENV: MANAGED_GUARD_FILE_ENV}.items()}


def validate_managed_project_guard(*, receipt: Mapping[str, Any] | None,
                                   secret_paths: Mapping[str, str], values: Mapping[str, str],
                                   stage_caps: Mapping[str, Any], openai_maximum_cost_usd: float) -> None:
    """Bind the observed hard limit to the selected project, key, and stage cap."""
    from .agent_execution.contracts import AgentExecutionError
    from .task_object_agents_api_stage import validate_managed_asset_guard

    policy = (receipt or {}).get("replacement_authoring_agents_api_policy")
    if not isinstance(policy, Mapping) or type(policy.get("ttl_seconds")) is not int:
        raise TaskEvaluationSceneConfigurationVastError("scene_configuration_agents_api_policy_missing")
    now_epoch = datetime.now(UTC).timestamp()
    try:
        validate_managed_asset_guard(policy=policy,
            guard_file=Path(secret_paths[MANAGED_GUARD_FILE_ENV]),
            project_id=values["OPENAI_PROJECT_ID"],
            credential_id=values["OPENAI_CONTENT_AGENTS_API_KEY_ID"],
            maximum_cost_usd=min(15.0, float(stage_caps["content_agents"]),
                                 float(openai_maximum_cost_usd)),
            deadline=now_epoch + policy["ttl_seconds"], now=now_epoch)
    except (AgentExecutionError, KeyError, TypeError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationVastError(
            "scene_configuration_agents_api_project_guard_invalid") from exc
