"""Network boundary of the site universe producer.

This is the only module in ``tools/site_universe`` that opens a network
connection. Everything downstream reads bytes from the raw cache.

The raw cache is a directory with one file per URL and a ``raw-manifest.json``
that records, for each URL, the retrieval time, the SHA-256 and the size of
the bytes. A cached file is reused unless ``refresh`` is set. The cache never
holds credentials: requests carry no authentication and no cookies.

Some public sources sit behind bot protection that refuses every non-browser
client. The producer does not work around that protection. A person can
download such a file in a browser and record it with :meth:`RawCache.import_file`,
which keeps the same provenance fields with ``acquisition = "manual_import"``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RAW_MANIFEST_NAME = "raw-manifest.json"
RAW_MANIFEST_SCHEMA = "blueprint.site_universe.raw_manifest.v1"
USER_AGENT = (
    "blueprint-site-universe/1 (bulk public-source snapshot; one cached request per file)"
)
DEFAULT_TIMEOUT_S = 300.0
MAX_BYTES = 2 * 1024 * 1024 * 1024
RETRY_STATUSES = frozenset({429, 502, 503, 504})
MAX_RETRIES = 2
CHUNK = 1024 * 1024


class FetchError(RuntimeError):
    """A raw input could not be obtained or failed its integrity check."""


class OfflineMiss(FetchError):
    """The URL is not cached and network access is disabled."""


class SourceBlocked(FetchError):
    """The publisher refused an automated client (for example HTTP 403)."""


@dataclass(frozen=True)
class RawEntry:
    url: str
    path: Path
    sha256: str
    bytes: int
    retrieved_at: str
    acquisition: str
    source_id: str
    content_type: str | None = None

    def read_bytes(self) -> bytes:
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise FetchError(f"raw cache file changed after it was recorded: {self.path}")
        return data

    def manifest_row(self) -> dict:
        return {
            "acquisition": self.acquisition,
            "bytes": self.bytes,
            "content_type": self.content_type,
            "path": self.path.name,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "source_id": self.source_id,
            "url": self.url,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_name(url: str, suffix: str = "") -> str:
    """Stable file name for a URL: a URL digest plus a readable suffix."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    if not suffix:
        tail = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        suffix = os.path.splitext(tail)[1].lower()
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    return digest + suffix


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


class RawCache:
    """Raw cache keyed by URL, with a manifest of retrieval time and SHA-256."""

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        allow_network: bool = True,
        refresh: bool = False,
        opener: Callable | None = None,
        clock: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ):
        self.root = Path(root)
        self.allow_network = allow_network
        self.refresh = refresh
        self._opener = opener or _default_opener
        self._clock = clock
        self._sleep = sleep
        self._log = log or (lambda message: None)
        self._last_request: dict[str, float] = {}
        self._refreshed: set[str] = set()
        self._manifest = self._load_manifest()

    # -- manifest -----------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / RAW_MANIFEST_NAME

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"schema": RAW_MANIFEST_SCHEMA, "entries": {}}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != RAW_MANIFEST_SCHEMA or not isinstance(
            manifest.get("entries"), dict
        ):
            raise FetchError(f"unrecognized raw manifest: {self.manifest_path}")
        return manifest

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        body = json.dumps(self._manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def _entry(self, url: str) -> RawEntry | None:
        row = self._manifest["entries"].get(url)
        if row is None:
            return None
        path = self.root / row["path"]
        if not path.exists():
            raise FetchError(f"raw manifest names a missing file: {path}")
        return RawEntry(
            url=url,
            path=path,
            sha256=row["sha256"],
            bytes=row["bytes"],
            retrieved_at=row["retrieved_at"],
            acquisition=row["acquisition"],
            source_id=row["source_id"],
            content_type=row.get("content_type"),
        )

    def entries(self) -> list[RawEntry]:
        return [self._entry(url) for url in sorted(self._manifest["entries"])]

    def cached(self, url: str) -> RawEntry | None:
        return self._entry(url)

    # -- acquisition ----------------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        source_id: str,
        suffix: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        min_interval_s: float = 0.0,
        validate: Callable[[Path], None] | None = None,
    ) -> RawEntry:
        """Return the cached entry for ``url``, downloading it when needed."""
        entry = self._entry(url)
        if entry is not None and not (self.refresh and url not in self._refreshed):
            if _file_sha256(entry.path) != entry.sha256:
                raise FetchError(f"raw cache file does not match its recorded SHA-256: {entry.path}")
            return entry
        if not self.allow_network:
            raise OfflineMiss(f"not in the raw cache and network access is disabled: {url}")
        entry = self._download(
            url,
            source_id=source_id,
            suffix=suffix,
            timeout_s=timeout_s,
            min_interval_s=min_interval_s,
            validate=validate,
        )
        self._refreshed.add(url)
        return entry

    def _download(self, url, *, source_id, suffix, timeout_s, min_interval_s, validate):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise FetchError(f"only https downloads are allowed: {url}")
        if parts.username or parts.password:
            raise FetchError("URLs with credentials are refused")
        host = parts.hostname or ""
        self.root.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            self._wait_for_host(host, min_interval_s)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Accept-Encoding": "identity"},
            )
            self._log(f"GET {url}")
            try:
                response = self._opener(request, timeout_s)
            except urllib.error.HTTPError as error:
                self._last_request[host] = time.monotonic()
                if error.code in RETRY_STATUSES and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = _retry_delay(error, attempt)
                    self._log(f"HTTP {error.code}; waiting {delay:.0f} s before retry {attempt}")
                    self._sleep(delay)
                    continue
                if error.code in (401, 403):
                    raise SourceBlocked(
                        f"HTTP {error.code}: the publisher refused an automated client for {url}"
                    ) from error
                raise FetchError(f"HTTP {error.code} for {url}") from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                self._last_request[host] = time.monotonic()
                raise FetchError(f"network failure for {url}: {error}") from error
            try:
                return self._store(url, response, source_id=source_id, suffix=suffix, validate=validate)
            finally:
                self._last_request[host] = time.monotonic()
                close = getattr(response, "close", None)
                if close:
                    close()

    def _store(self, url, response, *, source_id, suffix, validate):
        status = getattr(response, "status", 200)
        if status != 200:
            raise FetchError(f"HTTP {status} for {url}")
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("Content-Type") if hasattr(headers, "get") else None
        digest = hashlib.sha256()
        size = 0
        handle, temporary = tempfile.mkstemp(dir=self.root, prefix=".download-", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as out:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise FetchError(f"download exceeds {MAX_BYTES} bytes: {url}")
                    digest.update(chunk)
                    out.write(chunk)
            if size == 0:
                raise FetchError(f"empty response for {url}")
            if validate is not None:
                validate(Path(temporary))
            name = cache_name(url, suffix)
            os.replace(temporary, self.root / name)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        entry = RawEntry(
            url=url,
            path=self.root / name,
            sha256=digest.hexdigest(),
            bytes=size,
            retrieved_at=self._clock(),
            acquisition="network",
            source_id=source_id,
            content_type=content_type,
        )
        self._manifest["entries"][url] = entry.manifest_row()
        self._save_manifest()
        return entry

    def import_file(
        self,
        url: str,
        local_path: str | os.PathLike,
        *,
        source_id: str,
        retrieved_at: str | None = None,
        suffix: str = "",
        validate: Callable[[Path], None] | None = None,
    ) -> RawEntry:
        """Record a file that a person downloaded from ``url`` in a browser."""
        source = Path(local_path)
        if not source.is_file():
            raise FetchError(f"no such file to import: {source}")
        if validate is not None:
            validate(source)
        self.root.mkdir(parents=True, exist_ok=True)
        name = cache_name(url, suffix or source.suffix)
        target = self.root / name
        shutil.copyfile(source, target)
        sha256 = _file_sha256(target)
        entry = RawEntry(
            url=url,
            path=target,
            sha256=sha256,
            bytes=target.stat().st_size,
            retrieved_at=retrieved_at or self._clock(),
            acquisition="manual_import",
            source_id=source_id,
            content_type=None,
        )
        self._manifest["entries"][url] = entry.manifest_row()
        self._save_manifest()
        return entry

    def _wait_for_host(self, host: str, min_interval_s: float) -> None:
        if min_interval_s <= 0 or host not in self._last_request:
            return
        remaining = min_interval_s - (time.monotonic() - self._last_request[host])
        if remaining > 0:
            self._sleep(remaining)


def _retry_delay(error: urllib.error.HTTPError, attempt: int) -> float:
    header = error.headers.get("Retry-After") if error.headers else None
    if header and header.strip().isdigit():
        return min(float(header.strip()), 300.0)
    return 30.0 * (2 ** (attempt - 1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
