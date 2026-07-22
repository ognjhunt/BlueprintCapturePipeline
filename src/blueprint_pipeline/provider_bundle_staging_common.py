"""Provider-neutral token and URL primitives for bundle staging."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from .core.common import ensure_dir
from .secret_artifact_policy import redacted_secret_file_status


BUNDLE_ROUTE = "/bundle.zip"
OUTPUT_ROUTE = "/output.zip"
HEALTH_ROUTE = "/health"


def read_or_create_staging_token(path: Path) -> tuple[str, dict[str, Any]]:
    """Load or create a mode-0600 staging token without returning it in metadata."""

    ensure_dir(path.parent)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        created = False
    else:
        token = secrets.token_urlsafe(32)
        path.write_text(token + "\n", encoding="utf-8")
        created = True
    path.chmod(0o600)
    mode = oct(path.stat().st_mode & 0o777)
    status = redacted_secret_file_status(
        path,
        path_source="staging_token_file",
        raw_secret_field="token_recorded_in_manifest",
    )
    status.update(
        {
            "created": created,
            "present": path.is_file(),
            "mode": mode,
            "mode_is_0600": mode == "0o600",
            "token_recorded_in_manifest": False,
        }
    )
    return token, status


def staging_url_with_token(base_url: str, route: str, token: str) -> str:
    """Attach a staging token while normalizing the provider-neutral route."""

    parsed = urlparse(base_url)
    clean_path = "/" + route.strip("/")
    query = urlencode({"token": token})
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", query, ""))
