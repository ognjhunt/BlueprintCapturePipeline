"""Pure owner-attempt identity checks shared by retained evidence and paid admission."""
from __future__ import annotations

import re
from typing import Any, Mapping

SCHEMA = "task_evaluation_scene_attempt_binding.v1"
OWNER_FIELDS = {"scene_intent_digest", "scene_attempt_id", "scene_attempt_binding"}
POLICY_FIELDS = {"scene_policy_candidates", "scene_policy_binding"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class SceneExecutionAuthorityError(ValueError):
    pass

BINDING_FIELDS = {"schema_version", "intent_id", "intent_digest", "attempt_id", "source_commit",
                  "runtime_digest", "input_digest"}


def scene_execution_binding_blockers(value: Mapping[str, Any], *, source_commit: str | None = None) -> list[str]:
    if not (OWNER_FIELDS | POLICY_FIELDS).intersection(value):
        return []  # existing legacy authority still passes its original gates
    if not OWNER_FIELDS.issubset(value):
        return ["scene_execution_owner_binding_missing"]
    binding = value.get("scene_attempt_binding")
    required = BINDING_FIELDS
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
    return []


def require_scene_execution_binding(value: Mapping[str, Any], *, source_commit: str | None = None) -> None:
    blockers = scene_execution_binding_blockers(value, source_commit=source_commit)
    if blockers:
        raise SceneExecutionAuthorityError(",".join(blockers))
