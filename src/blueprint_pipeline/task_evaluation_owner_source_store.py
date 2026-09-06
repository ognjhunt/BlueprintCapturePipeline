"""Host-only transport for immutable, owner-provided scene source bytes.

The reserved URI namespace is a control-plane address, not an S3 object. It
fits the existing portable reference contract but is never sent to an object
store client. A missing local object fails closed; there is no network fallback.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

from .task_evaluation_scene_configuration_submission_inputs import checked_file, sha

PREFIX = "s3://blueprint/task-evaluation/host-only-owner-sources/"
ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_OWNER_SOURCE_STORE_ROOT"
DEFAULT_ROOT = "/var/lib/blueprint/task-evaluation-inputs/owner-source-store"
_RELATIVE = re.compile(r"([0-9a-f]{64})/([A-Za-z0-9][A-Za-z0-9._-]{0,180})\Z")


def source_uri(digest: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".usd", ".usda", ".usdc", ".glb", ".ply"}:
        raise ValueError("owner_source_format_not_supported")
    relative = digest.removeprefix("sha256:") + "/source" + suffix
    if not digest.startswith("sha256:") or not _RELATIVE.fullmatch(relative):
        raise ValueError("owner_source_reference_invalid")
    return PREFIX + relative


def source_path(uri: str, *, root: str | Path | None = None) -> Path:
    if not uri.startswith(PREFIX) or not _RELATIVE.fullmatch(uri[len(PREFIX):]):
        raise ValueError("owner_source_reference_invalid")
    base = Path(root or os.getenv(ROOT_ENV, DEFAULT_ROOT))
    path = base / uri[len(PREFIX):]
    if not base.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("owner_source_store_unsafe")
    return path


def install_source(*, source: Path, uri: str, digest: str, size_bytes: int,
                   root: str | Path | None = None) -> Path:
    """Publish one verified local object atomically without overwriting a writer."""
    target = source_path(uri, root=root)
    if uri.split("/")[-2] != digest.removeprefix("sha256:"):
        raise ValueError("owner_source_reference_digest_mismatch")
    reference = {"sha256": digest, "size_bytes": size_bytes}
    checked_file(source, reference)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if target.exists():
        return checked_file(target, reference)
    with tempfile.NamedTemporaryFile(prefix=".install-", dir=target.parent) as stream:
        with source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream, length=1024 * 1024)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
        checked_file(temporary, reference)
        os.fchmod(stream.fileno(), 0o440)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            pass  # A concurrent identical install is adopted only after rehash.
    return checked_file(target, reference)


def fetch_source(uri: str, destination: Path, maximum_bytes: int) -> None:
    path = source_path(uri)
    digest = "sha256:" + uri.split("/")[-2]
    checked_file(path, {"sha256": digest, "size_bytes": maximum_bytes})
    with path.open("rb") as incoming, destination.open("wb") as output:
        shutil.copyfileobj(incoming, output, length=1024 * 1024)
    if destination.stat().st_size != maximum_bytes or sha(destination) != digest:
        raise ValueError("owner_source_readback_mismatch")
