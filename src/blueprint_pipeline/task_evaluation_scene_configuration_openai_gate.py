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
# One configuration run reserves official OpenAI cost for up to three distinct
# ``paid_resource_class`` values, and every reservation demands a same-day
# zero-cost baseline for its exact ``(project_id, api_key_id)`` scope.  A single
# shared key/attestation therefore cannot serve two stages of one run: the
# second gate would refuse the attestation's class, and even with matching
# classes the first stage's spend would poison the second stage's baseline.
# Each stage binds its own exclusive key and its own operator attestation.
_STAGE_SCOPE_ENV = {
    "artifixer_semantic_teacher": {
        "api_key_file": "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_FILE",
        "api_key_id": "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
        "attestation_file": (
            "BLUEPRINT_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_COST_SCOPE_ATTESTATION_FILE"
        ),
    },
    "artifixer_visual_review": {
        "api_key_file": "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
        "api_key_id": "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID",
        "attestation_file": (
            "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE"
        ),
    },
    "content_agents": {
        "api_key_file": "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
        "api_key_id": "OPENAI_CONTENT_AGENTS_API_KEY_ID",
        "attestation_file": (
            "BLUEPRINT_OPENAI_CONTENT_AGENTS_COST_SCOPE_ATTESTATION_FILE"
        ),
    },
}
class TaskEvaluationSceneConfigurationOpenAIGateError(ValueError):
    """The parent run did not provide a valid bounded OpenAI authority."""


def scene_configuration_openai_stage_scope(
    environment: Mapping[str, str],
    *,
    stage: str,
) -> dict[str, str]:
    """Resolve one stage's exclusive OpenAI key/attestation scope, fail closed.

    There is deliberately no fallback to the shared single-scope environment
    names: reintroducing a shared scope re-creates the cross-stage attestation
    class mismatch and the nonzero same-day baseline collision this seam exists
    to prevent.
    """

    names = _STAGE_SCOPE_ENV.get(stage)
    if names is None:
        raise TaskEvaluationSceneConfigurationOpenAIGateError(
            "scene_configuration_openai_stage_unknown"
        )
    scope = {
        role: str(environment.get(name) or "").strip()
        for role, name in names.items()
    }
    if not all(scope.values()):
        raise TaskEvaluationSceneConfigurationOpenAIGateError(
            f"scene_configuration_openai_stage_scope_missing:{stage}"
        )
    return scope


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
    scope = scene_configuration_openai_stage_scope(environment, stage=stage)
    return build_openai_official_cost_run_gate(
        scope_attestation_path=scope["attestation_file"],
        admin_api_key_file=str(
            environment.get("OPENAI_ADMIN_API_KEY_FILE") or ""
        ),
        project_id=str(environment.get("OPENAI_PROJECT_ID") or ""),
        api_key_id=scope["api_key_id"],
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
    "scene_configuration_openai_stage_scope",
]
