"""Exact PyPI wheel bytes for provider bundles, downloaded once and kept by digest.

A provider worker imports what its bundle carries; it must not resolve a
package index on a rented GPU. Small wheels are vendored in ``_vendor``. A large
binary wheel does not belong in git, so the bundle builder names it by its
exact PyPI URL, size and SHA-256 and resolves it here, on the control plane,
before any provider is allocated:

* a cache entry is hashed again before every use;
* a miss streams the exact URL from ``files.pythonhosted.org`` and refuses any
  other host, size or digest before publishing the bytes under their digest;
* without a writable cache root the verified bytes go only to the caller's
  directory, for that one build.

The cache is reproducible: an evicted or damaged entry is fetched again.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SOURCE_ORIGIN = "https://files.pythonhosted.org"
CACHE_ROOT_ENV = "BLUEPRINT_PINNED_WHEEL_CACHE_ROOT"
DEFAULT_CACHE_PARENT = Path("/var/lib/blueprint/task-evaluation-inputs")
DEFAULT_CACHE_NAME = "pinned-wheels"
DOWNLOAD_TIMEOUT_SECONDS = 600
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")

Downloader = Callable[..., Any]


def resolve_cache_root(explicit: str | Path | None = None) -> Path | None:
    """The persistent cache root, or ``None`` when no writable one exists."""

    configured = str(explicit or os.environ.get(CACHE_ROOT_ENV) or "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif DEFAULT_CACHE_PARENT.is_dir() and os.access(DEFAULT_CACHE_PARENT, os.W_OK):
        root = DEFAULT_CACHE_PARENT / DEFAULT_CACHE_NAME
    else:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return root.resolve() if os.access(root, os.W_OK) and not root.is_symlink() else None


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    filename = str(row.get("wheel") or "")
    url = str(row.get("url") or "")
    size = row.get("size_bytes")
    digest = str(row.get("sha256") or "")
    if (
        not filename.endswith(".whl")
        or Path(filename).name != filename
        or not url.startswith(SOURCE_ORIGIN + "/")
        or not url.endswith("/" + filename)
        or type(size) is not int
        or size <= 0
        or _SHA256.fullmatch(digest) is None
    ):
        raise ValueError("pinned_wheel_identity_invalid")
    return filename, url, size, digest


def _matches(path: Path, *, size: int, digest: str) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest() == digest


def _download(url: str, *, output_path: Path, expected_sha256: str, max_bytes: int) -> None:
    from .safe_outbound_http import download_file, pinned_api_policy

    download_file(url, output_path=output_path, expected_sha256=expected_sha256, max_bytes=max_bytes,
                  timeout_seconds=DOWNLOAD_TIMEOUT_SECONDS,
                  policy=pinned_api_policy(SOURCE_ORIGIN, max_response_bytes=max_bytes))


def fetch_pinned_wheel(
    row: Mapping[str, Any],
    *,
    destination_dir: str | Path,
    cache_root: str | Path | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Return a local path holding exactly the pinned wheel bytes.

    ``row`` carries ``wheel`` (filename), ``url``, ``size_bytes`` and
    ``sha256``. The path is a cache entry, or ``destination_dir/<wheel>`` when
    no writable cache root exists.
    """

    filename, url, size, digest = _identity(row)
    cache = resolve_cache_root(cache_root)
    target = (cache / digest.removeprefix("sha256:") / filename if cache is not None
              else Path(destination_dir).expanduser().resolve() / filename)
    if _matches(target, size=size, digest=digest):
        return target
    if target.is_symlink() or target.exists():
        # A damaged or substituted entry is replaced, never trusted.
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        (downloader or _download)(url, output_path=target, expected_sha256=digest, max_bytes=size)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"pinned_wheel_download_failed:{filename}:{type(exc).__name__}") from exc
    if not _matches(target, size=size, digest=digest):
        target.unlink(missing_ok=True)
        raise ValueError(f"pinned_wheel_download_identity_mismatch:{filename}")
    return target


__all__ = [
    "CACHE_ROOT_ENV",
    "DEFAULT_CACHE_NAME",
    "DEFAULT_CACHE_PARENT",
    "SOURCE_ORIGIN",
    "fetch_pinned_wheel",
    "resolve_cache_root",
]
