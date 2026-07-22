"""Provider-neutral utilities shared by asynchronous WAM runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object, treating non-object JSON as an empty mapping."""

    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def redact_provider_url(value: str) -> str:
    """Retain a provider URL's origin/path while removing query credentials."""

    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return "<redacted-url>" if value else ""
    query = "REDACTED_QUERY" if parsed.query else ""
    fragment = "REDACTED_FRAGMENT" if parsed.fragment else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query, fragment))


def read_sensitive_url_file(
    path_value: str, *, label: str
) -> tuple[str, dict[str, Any]]:
    """Read a signed URL from a file while returning metadata, never the value."""

    if not str(path_value or "").strip():
        return "", {
            "label": label,
            "configured": False,
            "present": False,
            "raw_secret_values_recorded": False,
        }
    path = Path(path_value).expanduser().resolve()
    mode = oct(path.stat().st_mode & 0o777) if path.exists() else None
    try:
        value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except OSError as exc:
        return "", {
            "label": label,
            "configured": True,
            "path": str(path),
            "present": path.exists(),
            "mode": mode,
            "read_error": type(exc).__name__,
            "raw_secret_values_recorded": False,
        }
    return value, {
        "label": label,
        "configured": True,
        "path": str(path),
        "present": path.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "value_present": bool(value),
        "raw_secret_values_recorded": False,
    }
