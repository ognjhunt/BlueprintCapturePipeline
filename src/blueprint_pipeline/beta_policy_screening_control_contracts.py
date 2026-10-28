"""Pure exact screening control contracts, safe for the provider import closure."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math
import re
import time
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest

BINDING_FIELDS = frozenset({"campaign_id", "site_id", "plan_digest", "candidate_id", "configuration_digest",
    "attempt_id", "invocation_id", "prepared_digest", "authority_digest", "resource_name", "source"})
_SOURCE_FIELDS = frozenset({"scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
                           "task_context_digest", "scene_revision_digest"})
_DIGEST_FIELDS = frozenset({"plan_digest", "configuration_digest", "invocation_id", "prepared_digest", "authority_digest"})
_RESPONSE_FIELDS = frozenset({"schema_version", "binding", "binding_digest", "revision", "status",
    "cancellation_requested", "stop_reason", "execution_deadline_at_ms", "observed_at_iso", "next_check_at_ms",
    "new_execution_authorized", "read_only_closeout_allowed", "applicability", "control_digest"})


def validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if (set(binding) != BINDING_FIELDS or not isinstance(binding.get("source"), Mapping)
            or set(binding["source"]) != _SOURCE_FIELDS):
        raise ValueError("beta_screening_execution_control_binding_invalid")
    for key, value in binding.items():
        if key == "source":
            continue
        pattern = r"sha256:[0-9a-f]{64}" if key in _DIGEST_FIELDS else r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}"
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError("beta_screening_execution_control_binding_invalid")
    for key, value in binding["source"].items():
        pattern = r"sha256:[0-9a-f]{64}" if key.endswith("digest") else r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}"
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError("beta_screening_execution_control_binding_invalid")
    return dict(binding)


def validate_control(value: Mapping[str, Any], *, binding: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    binding = validate_binding(binding)
    moment = time.time() if now is None else now
    if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
        raise ValueError("beta_screening_execution_control_invalid")
    deadline, revision, next_check = value["execution_deadline_at_ms"], value["revision"], value["next_check_at_ms"]
    stopped = value["status"] == "stopped"
    try:
        observed = datetime.fromisoformat(value["observed_at_iso"].replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("missing timezone")
        observed_epoch = observed.timestamp()
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise ValueError("beta_screening_execution_control_invalid") from None
    if (value["schema_version"] != "blueprint.beta_policy_screening_execution_control.v1"
            or value["binding"] != binding or value["binding_digest"] != cross_runtime_canonical_digest(binding)
            or value["control_digest"] != cross_runtime_canonical_digest(value, digest_field="control_digest")
            or value["status"] not in {"current", "stopped"}
            or value["cancellation_requested"] is not stopped
            or (stopped and (not isinstance(value["stop_reason"], str)
                             or re.fullmatch(r"[a-z_]+", value["stop_reason"]) is None))
            or not stopped and value["stop_reason"] is not None
            or value["new_execution_authorized"] is not False or value["read_only_closeout_allowed"] is not True
            or value["applicability"] != ("historical_only" if stopped else "current")
            or any(type(item) is not int or not 0 < item <= 8_640_000_000_000_000
                   for item in (deadline, next_check))
            or type(revision) is not int or not 0 < revision <= 9_007_199_254_740_991
            or not math.isfinite(observed_epoch) or not 0 < observed_epoch <= moment + 60
            or not stopped and (moment - observed_epoch > 120 or deadline <= moment * 1000)
            or abs(next_check - (observed_epoch * 1000 + 60_000)) > 1):
        raise ValueError("beta_screening_execution_control_invalid")
    return dict(value)


