"""Derive exact episode rights from one human-approved run authority."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import math
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .episode_interpretation import (
    EpisodeInterpretationError,
    EpisodeInterpretationRequest,
    EpisodeInterpreter,
    materialize_episode_interpretation_rights,
)


SCHEMA_VERSION = "policy_canary_episode_interpretation_batch_authority.v1"
MANAGED_SCHEMA_VERSION = "policy_canary_episode_interpretation_batch_authority.v2"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
ALLOWED_ARTIFACT_ROLES = frozenset(
    {
        "task_success_contract",
        "deterministic_score",
        "state_trace",
        "contact_force_trace",
        "frame_manifest",
        "lossless_frame",
        "review_video",
    }
)


def validate_episode_interpretation_batch_authority_shape(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    authority = dict(value)
    allowed = authority.get("allowed_artifact_roles")
    identity = authority.get("interpreter")
    managed = authority.get("schema_version") == MANAGED_SCHEMA_VERSION
    if managed:
        policy = authority.get("agent_runtime_policy")
        if (not isinstance(policy, Mapping)
                or policy.get("budget_policy") != "project_guard_accepted_uncertainty"
                or policy.get("session_retention") != "until_deleted"
                or policy.get("trace_retention") != "provider_default" or policy.get("region") != "us"
                or not _DIGEST.fullmatch(str(policy.get("project_guard_receipt_digest") or ""))
                or policy.get("disclosure_scope") != "rights_admitted_episode_evidence"
                or type(authority.get("expires_at")) not in {float, int}
                or not math.isfinite(authority["expires_at"]) or time.time() >= authority["expires_at"]
                or type(authority.get("maximum_episodes")) is not int
                or not 1 <= authority["maximum_episodes"] <= 20):
            raise EpisodeInterpretationError("episode_interpretation_managed_batch_policy_invalid")
    if (
        authority.get("schema_version") not in {SCHEMA_VERSION, MANAGED_SCHEMA_VERSION}
        or authority.get("status") != "approved"
        or not str(authority.get("run_id") or "").strip()
        or not isinstance(identity, Mapping)
        or identity.get("principal_kind") != "independent_interpreter"
        or identity.get("provider_id") != "openai"
        or identity.get("execution_site") != "external_provider"
        or identity.get("runtime") != ("openai_agents_api" if managed else "openai_agents_sdk")
        or not str(identity.get("model") or "").strip()
        or not str(identity.get("model_version") or "").strip()
        or not _DIGEST.fullmatch(
            str(authority.get("interpreter_profile_digest") or "")
        )
        or not isinstance(allowed, list)
        or not set(allowed).issubset(ALLOWED_ARTIFACT_ROLES)
        or not allowed
        or authority.get("external_disclosure_authorized") is not True
        or authority.get("provider_training_authorized") is not False
        or authority.get("public_redistribution_authorized") is not False
        or not isinstance(authority.get("maximum_cost_usd"), (int, float))
        or isinstance(authority.get("maximum_cost_usd"), bool)
        or float(authority["maximum_cost_usd"]) <= 0
        or not math.isfinite(float(authority["maximum_cost_usd"]))
        or not _DIGEST.fullmatch(
            str(authority.get("source_rights_admission_digest") or "")
        )
        or not str(authority.get("accepted_by") or "").strip()
        or not str(authority.get("accepted_on") or "").strip()
        or not str(authority.get("authority_reference") or "").strip()
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise EpisodeInterpretationError(
            "episode_interpretation_batch_authority_invalid"
        )
    return authority


def validate_episode_interpretation_batch_authority(
    value: Mapping[str, Any],
    *,
    run_id: str,
    interpreter: EpisodeInterpreter,
    interpreter_profile_digest: str,
    maximum_cost_usd: float,
) -> dict[str, Any]:
    authority = validate_episode_interpretation_batch_authority_shape(value)
    identity = interpreter.identity
    if (
        authority.get("run_id") != run_id
        or authority.get("interpreter") != identity.__dict__
        or authority.get("interpreter_profile_digest") != interpreter_profile_digest
        or authority.get("maximum_cost_usd") != maximum_cost_usd
    ):
        raise EpisodeInterpretationError(
            "episode_interpretation_batch_authority_invalid"
        )
    return authority


def derive_episode_interpretation_rights(
    *,
    authority: Mapping[str, Any],
    request: EpisodeInterpretationRequest,
    interpreter: EpisodeInterpreter,
    output_path: str | Path,
) -> dict[str, Any]:
    disclosed = sorted(set(interpreter.disclosed_artifact_roles(request)))
    if any(role not in authority["allowed_artifact_roles"] for role in disclosed):
        raise EpisodeInterpretationError(
            "episode_interpretation_batch_authority_scope_exceeded"
        )
    destination = Path(output_path).expanduser().resolve()
    if destination.is_file() and not destination.is_symlink():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping):
            return dict(existing)
    return materialize_episode_interpretation_rights(
        episode_id=request.episode_id,
        input_bundle_digest=request.input_receipt["input_bundle_digest"],
        identity=interpreter.identity,
        allowed_artifact_roles=disclosed,
        external_disclosure_authorized=True,
        accepted_by=str(authority["accepted_by"]),
        accepted_on=str(authority["accepted_on"]),
        authority_reference=str(authority["authority_reference"]),
        source_rights_admission_digest=str(
            authority["source_rights_admission_digest"]
        ),
        output_path=destination,
        agent_runtime_policy=authority.get("agent_runtime_policy") if authority.get("schema_version") == MANAGED_SCHEMA_VERSION else None,
    )


__all__ = [
    "ALLOWED_ARTIFACT_ROLES",
    "SCHEMA_VERSION",
    "derive_episode_interpretation_rights",
    "validate_episode_interpretation_batch_authority",
    "validate_episode_interpretation_batch_authority_shape",
]
