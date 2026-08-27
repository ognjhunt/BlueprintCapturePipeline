"""Retain bounded, secret-clean evidence from the nested Content Agents runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import redacted_failure_text
from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_content_agents_failure_evidence.v1"
)
CONTENT_AGENTS_RUNTIME_ACCEPTED_STATUS = "completed"
_STREAM_TAIL_CHARS = 20_000
_MAX_MARKERS = 128
_MAX_MARKER_CHARS = 1_000
_MAX_BLOCKERS = 64
_MAX_BLOCKER_CHARS = 1_000
_MAX_RESULT_JSON_CHARS = 64_000
_MARKER_PREFIX = "BLUEPRINT_ADP_CONTENT_AGENTS_"
_PROVIDER_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".venv",
        ".ovrtx_venv",
        ".ovrtx_native_venv",
        ".ovphysx_venv",
        "content_agents_source",
    }
)
_SECRET_FILE_ENV_NAMES = (
    "OPENAI_API_KEY_FILE",
    "OPENAI_ADMIN_API_KEY_FILE",
    "OPENAI_CONTENT_AGENTS_API_KEY_FILE",
)


class ContentAgentsRuntimeFailureEvidenceError(RuntimeError):
    """The nested result was absent, invalid, or not accepted."""

    def __init__(self, detail: str, *, runtime_result_digest: str | None) -> None:
        super().__init__(detail)
        self.runtime_result_digest = runtime_result_digest


def _redact(value: Any, *, secrets: Sequence[str]) -> str:
    text = str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "REDACTED_SECRET")
    return redacted_failure_text(text)


def _tail(value: Any, *, secrets: Sequence[str]) -> dict[str, Any]:
    text = _redact(value, secrets=secrets)
    dropped = max(0, len(text) - _STREAM_TAIL_CHARS)
    return {
        "text": text[-_STREAM_TAIL_CHARS:],
        "earlier_character_count_dropped": dropped,
        "retained_character_count": min(len(text), _STREAM_TAIL_CHARS),
    }


def _markers(*streams: Any, secrets: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for stream in streams:
        for line in str(stream or "").splitlines():
            marker = line.strip()
            if not marker.startswith(_MARKER_PREFIX):
                continue
            rows.append(_redact(marker, secrets=secrets)[:_MAX_MARKER_CHARS])
    return rows[-_MAX_MARKERS:]


def _blockers(
    runtime_result: Mapping[str, Any] | None, *, secrets: Sequence[str]
) -> list[str]:
    value = runtime_result.get("blockers") if runtime_result is not None else None
    if not isinstance(value, list):
        return []
    return [
        _redact(item, secrets=secrets).strip()[:_MAX_BLOCKER_CHARS]
        for item in value[:_MAX_BLOCKERS]
        if isinstance(item, str) and _redact(item, secrets=secrets).strip()
    ]


def _runtime_result_copy(
    runtime_result: Mapping[str, Any] | None, *, secrets: Sequence[str]
) -> tuple[Any, bool]:
    if runtime_result is None:
        return None, False

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {_redact(key, secrets=secrets): scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            return _redact(value, secrets=secrets)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return _redact(value, secrets=secrets)

    copied = scrub(dict(runtime_result))
    encoded = canonical_json(copied)
    if len(encoded) <= _MAX_RESULT_JSON_CHARS:
        return copied, False
    return encoded[:_MAX_RESULT_JSON_CHARS], True


def failure_evidence_secret_values(
    environment: Mapping[str, str], *, known_values: Sequence[str] = ()
) -> tuple[str, ...]:
    """Read file-based secrets only so retained diagnostics can remove them."""

    values = [str(value) for value in known_values if str(value)]
    for name in _SECRET_FILE_ENV_NAMES:
        unresolved = str(environment.get(name) or "").strip()
        if not unresolved:
            continue
        path = Path(unresolved).expanduser()
        try:
            if path.is_symlink() or not path.is_file():
                continue
            secret = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if secret:
            values.append(secret)
    return tuple(dict.fromkeys(values))


def retain_content_agents_runtime_failure_evidence(
    *,
    destination: str | Path,
    completed: Any,
    runtime_result_path: str | Path,
    runtime_result: Mapping[str, Any] | None,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Seal the exact inner refusal outside provider-ZIP-excluded directories."""

    output = Path(destination).resolve()
    unresolved_source = Path(runtime_result_path).expanduser()
    source = unresolved_source.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("scene_configuration_content_agents_failure_evidence_exists")
    if not _PROVIDER_EXCLUDED_DIRECTORY_NAMES.isdisjoint(output.parts):
        raise ValueError(
            "scene_configuration_content_agents_failure_evidence_path_invalid"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    source_present = (
        not unresolved_source.is_symlink()
        and source.is_file()
        and not source.is_symlink()
    )
    secrets = tuple(str(value) for value in secret_values if str(value))
    blockers = _blockers(runtime_result, secrets=secrets)
    result_copy, result_truncated = _runtime_result_copy(
        runtime_result, secrets=secrets
    )
    returncode = getattr(completed, "returncode", None)
    detail = ";".join(blockers)
    if not detail:
        detail = (
            "runtime_result_missing"
            if runtime_result is None
            else f"runtime_returncode_{returncode}"
        )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "runtime_returncode": returncode,
        "runtime_result": result_copy,
        "runtime_result_present": runtime_result is not None,
        "runtime_result_source_path_recorded": False,
        "runtime_result_source_size_bytes": (
            source.stat().st_size if source_present else None
        ),
        "runtime_result_source_sha256": (
            "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
            if source_present
            else None
        ),
        "runtime_result_copy_truncated": result_truncated,
        "typed_blockers": blockers,
        "typed_markers": _markers(
            getattr(completed, "stdout", ""),
            getattr(completed, "stderr", ""),
            secrets=secrets,
        ),
        "stdout_tail": _tail(getattr(completed, "stdout", ""), secrets=secrets),
        "stderr_tail": _tail(getattr(completed, "stderr", ""), secrets=secrets),
        "raw_secret_values_recorded": False,
        "failure_detail": detail[:_MAX_BLOCKER_CHARS],
        "evidence_digest": "",
    }
    value["evidence_digest"] = canonical_digest(
        value, digest_field="evidence_digest"
    )
    output.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return value


def read_content_agents_runtime_result(
    *,
    completed: Any,
    runtime_result_path: str | Path,
    evidence_path: str | Path,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Return an accepted result or retain the exact refusal before raising."""

    path = Path(runtime_result_path).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        retain_content_agents_runtime_failure_evidence(
            destination=evidence_path,
            completed=completed,
            runtime_result_path=path,
            runtime_result=None,
            secret_values=secret_values,
        )
        raise ContentAgentsRuntimeFailureEvidenceError(
            "scene_configuration_content_agents_runtime_result_missing",
            runtime_result_digest=None,
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        retain_content_agents_runtime_failure_evidence(
            destination=evidence_path,
            completed=completed,
            runtime_result_path=path,
            runtime_result=None,
            secret_values=secret_values,
        )
        raise ContentAgentsRuntimeFailureEvidenceError(
            "scene_configuration_content_agents_runtime_result_missing",
            runtime_result_digest=None,
        )
    runtime_result = dict(value)
    result_digest = str(runtime_result.get("result_digest") or "") or None
    if completed.returncode == 0 and runtime_result.get("status") == (
        CONTENT_AGENTS_RUNTIME_ACCEPTED_STATUS
    ):
        return runtime_result
    evidence = retain_content_agents_runtime_failure_evidence(
        destination=evidence_path,
        completed=completed,
        runtime_result_path=path,
        runtime_result=runtime_result,
        secret_values=secret_values,
    )
    raise ContentAgentsRuntimeFailureEvidenceError(
        "scene_configuration_content_agents_runtime_failed:"
        + str(evidence["failure_detail"]),
        runtime_result_digest=result_digest,
    )


__all__ = [
    "CONTENT_AGENTS_RUNTIME_ACCEPTED_STATUS",
    "ContentAgentsRuntimeFailureEvidenceError",
    "SCHEMA_VERSION",
    "failure_evidence_secret_values",
    "read_content_agents_runtime_result",
    "retain_content_agents_runtime_failure_evidence",
]
