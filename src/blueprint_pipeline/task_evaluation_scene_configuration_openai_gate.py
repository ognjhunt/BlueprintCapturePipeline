"""Official-cost gate for OpenAI calls inside one scene-configuration stage."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

from .openai_official_cost_gate import (
    OpenAIOfficialCostRunGate,
    build_openai_official_cost_run_gate,
)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_STAGE_CAP_ENV = {
    "artifixer_semantic_teacher": (
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD"
    ),
    "artifixer_visual_review": (
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD"
    ),
    "content_agents": (
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"
    ),
}


class TaskEvaluationSceneConfigurationOpenAIGateError(ValueError):
    """The parent run did not provide a valid bounded OpenAI authority."""


def scene_configuration_openai_stage_gate(
    *,
    environment: Mapping[str, str],
    stage: str,
    run_id: str,
    request_digest: str,
    candidate_digest: str,
    output_root: str | Path,
) -> OpenAIOfficialCostRunGate:
    """Build but do not reserve one exact stage's official provider gate."""

    cap_env = _STAGE_CAP_ENV.get(stage)
    try:
        stage_cap = float(environment.get(str(cap_env)) or 0)
        total_cap = float(
            environment.get("BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD")
            or 0
        )
        maximum_requests = int(
            environment.get("BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS")
            or 0
        )
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationSceneConfigurationOpenAIGateError(
            "scene_configuration_openai_authority_invalid"
        ) from exc
    authority_digest = str(
        environment.get("BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST") or ""
    )
    if (
        cap_env is None
        or not math.isfinite(stage_cap)
        or not math.isfinite(total_cap)
        or stage_cap <= 0
        or total_cap < stage_cap
        or maximum_requests <= 0
        or _DIGEST.fullmatch(authority_digest) is None
        or _DIGEST.fullmatch(request_digest) is None
        or _DIGEST.fullmatch(candidate_digest) is None
    ):
        raise TaskEvaluationSceneConfigurationOpenAIGateError(
            "scene_configuration_openai_authority_invalid"
        )
    return build_openai_official_cost_run_gate(
        scope_attestation_path=str(
            environment.get("BLUEPRINT_OPENAI_COST_SCOPE_ATTESTATION_FILE") or ""
        ),
        admin_api_key_file=str(
            environment.get("OPENAI_ADMIN_API_KEY_FILE") or ""
        ),
        project_id=str(environment.get("OPENAI_PROJECT_ID") or ""),
        api_key_id=str(environment.get("OPENAI_API_KEY_ID") or ""),
        lane_id=f"task_evaluation_scene_configuration_{stage}",
        run_id=run_id,
        request_digest=request_digest,
        candidate_digest=candidate_digest,
        authorization_receipt_digest=authority_digest,
        max_cost_usd=stage_cap,
        output_root=output_root,
        provider_id="openai",
        paid_resource_class=f"task_evaluation_scene_configuration_{stage}",
    )


__all__ = [
    "TaskEvaluationSceneConfigurationOpenAIGateError",
    "scene_configuration_openai_stage_gate",
]
