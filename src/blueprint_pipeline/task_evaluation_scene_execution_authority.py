"""Reopen persistent owner authority immediately before a new paid admission.

This supplements, never replaces, canonical allocator/standing authority. It is
not applied to billing, retained-output delivery, or teardown of existing runs.
"""

from __future__ import annotations

import json
import math
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .task_evaluation_scene_intake import CLIENTS_ENV, ROOT_ENV, SceneIntakeError, _read

from .task_evaluation_scene_attempt_binding import (
    BINDING_FIELDS, OWNER_FIELDS, POLICY_FIELDS, SCHEMA, SceneExecutionAuthorityError,
    scene_execution_binding_blockers,
)


def bind_scene_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    binding = {key: attempt[key] for key in ("intent_id", "intent_digest", "attempt_id", "source_commit",
                                            "runtime_digest", "input_digest")}
    return {"scene_intent_digest": binding["intent_digest"], "scene_attempt_id": binding["attempt_id"],
            "scene_attempt_binding": {"schema_version": SCHEMA, **binding}}


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def scene_execution_authority_blockers(
    value: Mapping[str, Any], *, source_commit: str | None = None,
    maximum_spend_usd: float | None = None, reopen_records: bool = True,
    provider: str | None = None,
    queue_root: str | Path | None = None, now: float | None = None,
) -> list[str]:
    blockers = scene_execution_binding_blockers(value, source_commit=source_commit)
    if blockers or not (OWNER_FIELDS | POLICY_FIELDS).intersection(value) or not reopen_records:
        return blockers
    binding, required = value['scene_attempt_binding'], BINDING_FIELDS
    configured = str(queue_root or os.getenv(ROOT_ENV, ""))
    if not configured:
        return ["scene_execution_owner_store_missing"]
    root = Path(configured)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        return ["scene_execution_owner_store_unsafe"]
    directory = root / binding["intent_id"]
    try:
        if directory.is_symlink() or (directory / "attempts").is_symlink():
            return ["scene_execution_owner_store_unsafe"]
        intent = _read(directory / "intent.json", "intent_digest")
        attempt = _read(directory / "attempts" / (binding["attempt_id"] + ".json"), "attempt_digest")
    except (SceneIntakeError, OSError, ValueError, TypeError):
        return ["scene_execution_owner_record_invalid"]
    if (intent.get("intent_digest") != binding["intent_digest"]
            or any(attempt.get(k) != binding[k] for k in required - {"schema_version"})):
        return ["scene_execution_owner_record_mismatch"]
    from .task_evaluation_scene_execution_budget import validate_attempt_execution_budget
    try:
        validate_attempt_execution_budget(directory, intent, attempt)
    except (ValueError, OSError, KeyError, TypeError):
        return ["scene_execution_owner_budget_extension_invalid"]
    from .task_evaluation_retained_controls_evidence import validated_cancellation
    try:
        if validated_cancellation(directory, attempt) is not None:
            return ["scene_execution_owner_attempt_cancelled_before_execution"]
    except (ValueError, OSError):
        return ["scene_execution_owner_cancellation_invalid"]
    trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
    if intent.get("authenticated_issuer") not in trusted:
        return ["scene_execution_owner_issuer_not_authorized"]
    if (directory / "revoked.json").exists():
        return ["scene_execution_owner_revoked"]
    request = intent.get("request", {})
    execution, consent = request.get("execution", {}), request.get("consent", {})
    actual_provider = provider or value.get("provider")
    if not actual_provider:
        argv = value.get("allocator", {}).get("argv", [])
        if isinstance(argv, list) and argv.count("--provider") == 1 and argv.index("--provider") + 1 < len(argv):
            actual_provider = argv[argv.index("--provider") + 1]
    if actual_provider != attempt.get("provider"):
        return ["scene_execution_owner_provider_mismatch"]
    moment = time.time() if now is None else now
    from .task_evaluation_scene_execution_window import effective_execution_expiry
    try:
        expiry = effective_execution_expiry(directory, intent)
    except (ValueError, OSError, TypeError):
        return ["scene_execution_owner_window_invalid"]
    if not _positive(expiry) or moment >= expiry:
        return ["scene_execution_owner_expired"]
    correction = attempt.get('visual_review_correction')
    if correction is not None:
        from .task_evaluation_visual_review_authority import read_authority
        try:
            grant = read_authority(directory=directory,source_attempt_id=correction['source_attempt_id'],admission=True,now=moment)
            if (grant is None or grant['authority_digest'] != correction.get('authority_digest')
                    or correction.get('scope') != 'placement_visual_review_only'
                    or actual_provider != 'openai' or attempt['maximum_spend_usd'] > grant['maximum_cost_usd']):
                return ['scene_execution_visual_review_correction_invalid']
        except (ValueError,OSError,KeyError,TypeError):
            return ['scene_execution_visual_review_correction_invalid']
    if (consent.get("spend_authorized") is not True or consent.get("task_confirmed") is not True
            or consent.get("private_processing_authorized") is not True
            or consent.get("provider_training_authorized") is not False
            or consent.get("accepted_by") != request.get("owner", {}).get("user_id")
            or attempt.get("provider") not in execution.get("allowed_providers", [])):
        return ["scene_execution_owner_consent_invalid"]
    spend = maximum_spend_usd
    if spend is None:
        spend = value.get("allocator", {}).get("max_spend_usd")
    reserved = attempt.get("maximum_spend_usd")
    if not _positive(spend) or not _positive(reserved) or Decimal(str(spend)) > Decimal(str(reserved)):
        return ["scene_execution_owner_reservation_insufficient"]
    if "scene_policy_candidates" in value:
        try:
            supplied = {row["id"]: row["artifact_digest"] for row in value["scene_policy_candidates"]}
            expected = {row["id"]: row["artifact_digest"] for row in execution["policy_candidates"]}
            if len(value["scene_policy_candidates"]) != 2 or len(supplied) != 2 or supplied != expected:
                return ["scene_execution_owner_policy_pair_mismatch"]
        except (KeyError, TypeError):
            return ["scene_execution_owner_policy_pair_mismatch"]
    return []


def require_scene_execution_authority(value: Mapping[str, Any], **kwargs: Any) -> None:
    blockers = scene_execution_authority_blockers(value, **kwargs)
    if blockers:
        raise SceneExecutionAuthorityError(",".join(blockers))


def validate_policy_setup_owner_fields(setup: Mapping[str, Any]) -> None:
    """Structural/spec checks apply to both new runs and retained delivery."""
    if not (OWNER_FIELDS | POLICY_FIELDS).intersection(setup):
        return
    require_scene_execution_authority(setup, reopen_records=False)
    from .task_evaluation_scene_policy_binding import execution_setup_binding_blockers
    specs = [json.loads(Path(setup["records"][name]["path"]).read_text())
             for name in ("pi05_execution_spec", "groot_execution_spec")]
    blockers = execution_setup_binding_blockers(setup, specs)
    if blockers:
        raise SceneExecutionAuthorityError(",".join(blockers))


def proves_no_provider_allocation(adapter: Mapping[str, Any]) -> bool:
    """Preserved legacy canary closeout predicate, not a new allocation grant."""
    instance_ids = adapter.get("vast_instance_ids")
    return bool(
        instance_ids in (None, [])
        and adapter.get("provider_mutations_performed") in {0, False}
        and adapter.get("provider_create_attempted") is not True
        and adapter.get("vast_side_effects_may_have_occurred") is not True
        and adapter.get("continuing_spend_from_this_run") is not True
    )
