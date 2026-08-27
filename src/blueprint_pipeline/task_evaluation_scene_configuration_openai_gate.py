"""Official-cost gate for OpenAI calls inside one scene-configuration stage."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .openai_official_cost_gate import (
    OpenAIOfficialCostRunGate,
    build_openai_official_cost_run_gate,
)
from .task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    derive_operator_scope_attestation,
    validate_openai_cost_scope_attestation,
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



# A derived receipt authorizes only the window it is used in. The operator's own
# file may carry a multi-week window; a receipt this lane mints for itself gets
# a short one so a stale copy left on a host cannot authorize spend days later.
_DERIVED_SCOPE_WINDOW = timedelta(hours=12)
_DERIVED_SCOPE_BACKDATE = timedelta(hours=1)


def stage_paid_resource_class(stage: str) -> str:
    """The one class name this lane spends under for ``stage``."""

    return f"task_evaluation_scene_configuration_{stage}"


def resolve_stage_scope_attestation(
    *,
    attestation: Mapping[str, Any] | None,
    paid_resource_class: str,
    project_id: str,
    api_key_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a valid scope receipt for this stage, deriving one if needed.

    An operator-written file is honoured whenever it validates. When there is
    no file, or the file predates a class rename, the lane derives an
    equivalent receipt from the operator-provisioned key binding rather than
    refusing: the caller has already proven that this key is provisioned for
    this stage and shared with no sibling stage, which is the whole of the
    exclusivity this receipt asserts.
    """

    if attestation is not None:
        try:
            return validate_openai_cost_scope_attestation(
                attestation,
                provider_id="openai",
                paid_resource_class=paid_resource_class,
                project_id=project_id,
                api_key_id=api_key_id,
            )
        except OpenAICostAuthorityError:
            pass
    moment = now or datetime.now(UTC)
    operator_id = str(
        (attestation or {}).get("operator_id")
        or os.environ.get("BLUEPRINT_OPENAI_SCOPE_OPERATOR_ID")
        or ""
    ).strip() or f"operator_scope_binding:{project_id}"
    derived = derive_operator_scope_attestation(
        provider_id="openai",
        paid_resource_class=paid_resource_class,
        project_id=project_id,
        api_key_id=api_key_id,
        operator_id=operator_id,
        exclusive_from=moment - _DERIVED_SCOPE_BACKDATE,
        exclusive_until=moment + _DERIVED_SCOPE_WINDOW,
    )
    return validate_openai_cost_scope_attestation(
        derived,
        provider_id="openai",
        paid_resource_class=paid_resource_class,
        project_id=project_id,
        api_key_id=api_key_id,
    )


def read_stage_scope_attestation(path: str | Path) -> dict[str, Any] | None:
    """Read an operator receipt, treating an unusable file as simply absent."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def materialize_stage_scope_attestation(
    environment: Mapping[str, str],
    *,
    stage: str,
    output_root: str | Path,
) -> Path:
    """Resolve this stage's receipt and write the exact bytes the gate reads."""

    scope = scene_configuration_openai_stage_scope(environment, stage=stage)
    resolved = resolve_stage_scope_attestation(
        attestation=read_stage_scope_attestation(scope["attestation_file"]),
        paid_resource_class=stage_paid_resource_class(stage),
        project_id=str(environment.get("OPENAI_PROJECT_ID") or ""),
        api_key_id=scope["api_key_id"],
    )
    destination = Path(output_root) / f"openai_cost_scope_attestation_{stage}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination.chmod(0o600)
    return destination


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
        scope_attestation_path=materialize_stage_scope_attestation(
            environment, stage=stage, output_root=output_root
        ),
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
        # This lane records the pre-call baseline and is charged the delta, so
        # a stage key that already spent earlier in the UTC day can still be
        # measured exactly. Requiring zero made each stage scope usable once
        # per day: scene 839873's semantic teacher billed $0.877128 on
        # 2026-08-27 and every later attempt that day was refused before it
        # could reserve, with nothing to show for the money already spent.
        require_zero_baseline=False,
    )


__all__ = [
    "TaskEvaluationSceneConfigurationOpenAIGateError",
    "materialize_stage_scope_attestation",
    "read_stage_scope_attestation",
    "resolve_stage_scope_attestation",
    "stage_paid_resource_class",
    "scene_configuration_openai_stage_gate",
    "scene_configuration_openai_stage_scope",
]
