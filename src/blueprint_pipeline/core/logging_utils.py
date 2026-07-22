"""Provider-neutral structured logging helpers for Blueprint paths."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_RESERVED_LOG_RECORD_KEYS = set(
    logging.LogRecord(
        name="blueprint",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
)
_SENSITIVE_FIELD_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTHORIZATION",
    "SIGNATURE",
)


def _redacted_marker(key: str) -> str:
    normalized = key.lower().replace("_", "-")
    return f"<redacted:{normalized or 'sensitive'}>"


def _sanitize_value(key: str, value: Any) -> Any:
    key_upper = key.upper()
    if any(marker in key_upper for marker in _SENSITIVE_FIELD_MARKERS):
        return _redacted_marker(key)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        lower_value = value.lower()
        if "x-goog-signature=" in lower_value or "signature=" in lower_value:
            return "<redacted:signed-url>"
        return value
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_value(str(item_key), item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_sanitize_value(key, item) for item in value]
    return value


def _format_fields(fields: Mapping[str, Any]) -> str:
    if not fields:
        return ""
    parts = [f"{key}={fields[key]!r}" for key in sorted(fields)]
    return " " + " ".join(parts)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Emit a stable Blueprint event with queryable LogRecord fields.

    The standard library logger is used to keep this repo dependency-light. The
    event name and sanitized field map are attached to the LogRecord as
    ``blueprint_event`` and ``blueprint_fields`` while safe scalar field names
    are also exposed directly for tests and log collectors that preserve
    ``extra`` attributes.
    """

    clean_fields = {
        str(key): _sanitize_value(str(key), value)
        for key, value in fields.items()
        if value is not None
    }
    extra: dict[str, Any] = {
        "blueprint_event": event,
        "blueprint_fields": clean_fields,
    }
    for key, value in clean_fields.items():
        if key.isidentifier() and key not in _RESERVED_LOG_RECORD_KEYS and key not in extra:
            extra[key] = value
    logger.log(level, (message or event) + _format_fields(clean_fields), extra=extra)
