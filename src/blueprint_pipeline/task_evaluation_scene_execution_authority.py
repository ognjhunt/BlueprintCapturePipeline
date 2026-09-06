"""Reopen persistent owner authority immediately before a new paid admission.

This supplements, never replaces, canonical allocator/standing authority. It is
not applied to billing, retained-output delivery, or teardown of existing runs.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .task_evaluation_scene_intake import CLIENTS_ENV, ROOT_ENV, SceneIntakeError, _read

SCHEMA = "task_evaluation_scene_attempt_binding.v1"
OWNER_FIELDS = {"scene_intent_digest", "scene_attempt_id", "scene_attempt_binding"}
POLICY_FIELDS = {"scene_policy_candidates", "scene_policy_binding"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SceneExecutionAuthorityError(ValueError):
    pass


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
    if not (OWNER_FIELDS | POLICY_FIELDS).intersection(value):
        return []  # existing legacy authority still passes its original gates
    if not OWNER_FIELDS.issubset(value):
        return ["scene_execution_owner_binding_missing"]
    binding = value.get("scene_attempt_binding")
    required = {"schema_version", "intent_id", "intent_digest", "attempt_id", "source_commit",
                "runtime_digest", "input_digest"}
    if (not isinstance(binding, Mapping) or set(binding) != required or binding.get("schema_version") != SCHEMA
            or any(not isinstance(binding.get(k), str) or _ID.fullmatch(binding[k]) is None
                   for k in ("intent_id", "attempt_id"))
            or any(not isinstance(binding.get(k), str) or _DIGEST.fullmatch(binding[k]) is None
                   for k in ("intent_digest", "runtime_digest", "input_digest"))
            or binding.get("intent_digest") != value.get("scene_intent_digest")
            or binding.get("attempt_id") != value.get("scene_attempt_id")
            or not re.fullmatch(r"[0-9a-f]{40}", str(binding.get("source_commit")))
            or binding["source_commit"] != (source_commit or value.get("source_commit"))):
        return ["scene_execution_owner_binding_invalid"]
    if not reopen_records:
        return []
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
    expiry = execution.get("expires_at_epoch")
    if not _positive(expiry) or moment >= expiry:
        return ["scene_execution_owner_expired"]
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
