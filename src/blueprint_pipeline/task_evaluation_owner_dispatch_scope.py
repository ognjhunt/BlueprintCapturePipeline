"""Select persistent-owner queue rows without consuming paused legacy work.

Selection grants no execution authority. Dispatch still reopens the complete
owner reservation, frozen profile, standing grant, release and provider gates.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_execution_authority import OWNER_FIELDS
from .task_evaluation_scene_configuration_submission_inputs import sha

SCOPE_ENV = "BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE"


def owner_scope_enabled():
    value = os.getenv(SCOPE_ENV, "all_authorized")
    if value not in {"all_authorized", "persistent_owner_only"}:
        raise ValueError("dispatch_owner_scope_invalid")
    return value == "persistent_owner_only"


def _read(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("dispatch_scope_path_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("dispatch_scope_record_invalid")
    return value


def _owned(value):
    return re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("scene_intent_digest", ""))) is not None


def owner_dispatch_blockers(value):
    return (["dispatch_persistent_owner_binding_required"] if owner_scope_enabled()
            and (not _owned(value) or not OWNER_FIELDS.issubset(value)) else [])


def require_owner_dispatch_scope(value):
    blockers = owner_dispatch_blockers(value)
    if blockers:
        raise ValueError(blockers[0])


def owner_launch_sources(sources, profile_dir):
    if not owner_scope_enabled():
        return sources
    selected = []
    for path in sources:
        try:
            request = _read(path)
            identifier = request.get("launch_profile_id", "")
            if not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", identifier) is None:
                continue
            profile = _read(Path(profile_dir) / (identifier + ".json"))
            if (_owned(profile) and OWNER_FIELDS.issubset(profile)
                    and request.get("launch_profile_digest") == profile.get("profile_digest")):
                selected.append(path)
        except (OSError, ValueError, TypeError):
            continue  # Unknown ownership is never permission to consume a row.
    return selected


def owner_policy_sources(sources):
    if not owner_scope_enabled():
        return sources
    selected = []
    for path in sources:
        try:
            envelope = _read(path)
            if (_owned(envelope) and envelope.get("envelope_digest") ==
                    canonical_digest(envelope, digest_field="envelope_digest")):
                selected.append(path)
        except (OSError, ValueError, TypeError):
            continue
    return selected


def policy_dispatch_envelope(result, result_path):
    envelope = {"schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": result["activation_id"], "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution", "source_commit": result["source_commit"],
        "activation_result": {"path": str(result_path), "size_bytes": result_path.stat().st_size,
                              "sha256": sha(result_path)},
        **{key: result[key] for key in ("capture_session_id", "intake_id", "task_success_contract", "task_success_contract_digest")},
        "request_digest": result["website_request_digest"], "maximum_provider_allocations": 1,
        "retry_cap": 0, "automatic_retry_authorized": False, "provider_mutation_performed": False,
        "paid_execution_requested": False}
    if "scene_intent_digest" in result:
        envelope["scene_intent_digest"] = result["scene_intent_digest"]
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    return envelope
