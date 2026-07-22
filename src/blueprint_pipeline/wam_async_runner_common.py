"""Provider-neutral utilities shared by asynchronous WAM runners."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse, urlsplit, urlunparse


@dataclass(frozen=True)
class AsyncPollDeadline:
    """Provider-neutral monotonic deadline and retry scheduling contract."""

    started_monotonic: float
    deadline_monotonic: float
    max_wait_seconds: int
    retry_interval_seconds: int

    @classmethod
    def start(
        cls,
        *,
        max_wait_seconds: int,
        retry_interval_seconds: int,
        started_monotonic: float | None = None,
        deadline_base_monotonic: float | None = None,
    ) -> "AsyncPollDeadline":
        if started_monotonic is None:
            started = time.monotonic()
            deadline_base = (
                time.monotonic()
                if deadline_base_monotonic is None
                else deadline_base_monotonic
            )
        else:
            started = started_monotonic
            deadline_base = (
                started
                if deadline_base_monotonic is None
                else deadline_base_monotonic
            )
        bounded_wait = max(0, int(max_wait_seconds))
        return cls(
            started_monotonic=started,
            deadline_monotonic=deadline_base + bounded_wait,
            max_wait_seconds=bounded_wait,
            retry_interval_seconds=max(1, int(retry_interval_seconds)),
        )

    def is_open(self, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return now <= self.deadline_monotonic

    def can_retry(self, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return now + self.retry_interval_seconds <= self.deadline_monotonic

    def wait_for_retry(
        self,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        (sleeper or time.sleep)(self.retry_interval_seconds)

    def elapsed_seconds(self, now_monotonic: float | None = None) -> float:
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return max(0.0, now - self.started_monotonic)

    def expired(self, now_monotonic: float | None = None) -> bool:
        return self.elapsed_seconds(now_monotonic) >= self.max_wait_seconds


def deadline_capped_wait_seconds(
    *,
    state: Mapping[str, Any],
    requested_max_wait_seconds: int,
    now_epoch: float,
) -> tuple[int, float | None, bool]:
    """Cap one provider poll wait to its persisted paid-resource deadline."""

    requested = max(0, int(requested_max_wait_seconds))
    raw_deadline = state.get("max_live_deadline_epoch")
    if isinstance(raw_deadline, bool):
        deadline_epoch = 0.0
    else:
        try:
            deadline_epoch = float(raw_deadline or 0.0)
        except (TypeError, ValueError):
            deadline_epoch = 0.0
    if deadline_epoch <= 0.0:
        return requested, None, False
    seconds_until_deadline = deadline_epoch - now_epoch
    capped = min(requested, max(0, int(seconds_until_deadline)))
    return capped, seconds_until_deadline, capped < requested


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
