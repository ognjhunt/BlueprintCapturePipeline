"""Provider-neutral utilities shared by asynchronous WAM runners."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit
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


def download_url_to_file(
    *,
    url: str,
    output_path: Path,
    user_agent: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Download one provider artifact without recording its signed source URL."""

    try:
        parsed_url = urlsplit(url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or parsed_url.username
            or parsed_url.password
        ):
            raise ValueError("provider_artifact_url_must_be_credential_free_https")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            data = response.read()
            http_status = int(getattr(response, "status", 200))
        output_path.write_bytes(data)
        return {
            "status": "completed",
            "http_status_code": http_status,
            "downloaded_size_bytes": len(data),
            "output_present": output_path.is_file(),
            "raw_secret_values_recorded": False,
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "http_error",
            "http_status_code": exc.code,
            "error_type": "HTTPError",
            "output_present": output_path.is_file(),
            "raw_secret_values_recorded": False,
        }
    except Exception as exc:  # pragma: no cover - provider/network diagnostics.
        return {
            "status": "blocked",
            "error_type": type(exc).__name__,
            "output_present": output_path.is_file(),
            "raw_secret_values_recorded": False,
        }
