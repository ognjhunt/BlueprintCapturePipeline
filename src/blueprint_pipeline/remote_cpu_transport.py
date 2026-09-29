"""Stage a remote CPU job's inputs and its release source in B2 by digest (plan 14 §4, §5).

Every staged object is content addressed: ``…/artifacts/remote-cpu-input/sha256/<hex>/input.bin``
for an input and ``…/artifacts/remote-cpu-source/sha256/<hex>/source.tar`` for the release source,
so a digest is staged once however many rows or attempts name it.  A HEAD by digest comes first;
only a miss publishes, through the configured-scene publisher, which reads back every byte it
uploaded.  An object already at the key with another identity is refused, never overwritten.

Nothing is written to host disk.  An input streams from the file the host already holds, and the
release source streams from ``git archive`` twice: once to hash it and, only when the HEAD misses,
once more to upload it.  The archive is recipe v2 (``git_archive_tar.v2``): ``src``,
``docs/schemas`` and ``pyproject.toml``, because the compile reads schemas relative to the
repository root (plan 14 C1).  The worker re-hashes every byte it fetches, so a HEAD hit trusts
the digest the caller verified (eligibility verifies every reference, plan 14 §13).
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .remote_cpu_job_contract import SOURCE_ARCHIVE_PATHS, SOURCE_ARCHIVE_RECIPE
from .task_evaluation_configured_scene_object_store import (
    LARGE_ARTIFACT_KEY_PREFIX,
    TaskEvaluationConfiguredSceneObjectStoreError,
    _object_missing,
    publish_configured_scene_stream,
)

INPUT_KIND, INPUT_FILENAME = "remote-cpu-input", "input.bin"
SOURCE_KIND, SOURCE_FILENAME = "remote-cpu-source", "source.tar"
GIT_TIMEOUT_SECONDS = 120
# git itself takes about a second; the upload pass streams through the same pipe, so this bounds the upload too.
ARCHIVE_TIMEOUT_SECONDS = 900
_CHUNK = 1024 * 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
# ``archive(repository, source_commit, sink)`` writes the release archive's bytes into ``sink``.
Archive = Callable[[Path, str, Any], None]


class RemoteCpuTransportError(RuntimeError):
    """A typed refusal to stage; ``code`` never carries a URL, a key or a provider message."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def cas_key(kind: str, digest: str, filename: str) -> str:
    """The content-addressed key of one staged object."""

    return f"{LARGE_ARTIFACT_KEY_PREFIX}/{kind}/sha256/{digest.removeprefix('sha256:')}/{filename}"


def _staged(client: Any, bucket: str, key: str, digest: str, size: int) -> bool:
    """HEAD by digest: ``True`` when the object is there with this identity, ``False`` when it is absent."""

    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(exc):
            return False
        raise RemoteCpuTransportError(f"remote_cpu_cas_head_failed:{type(exc).__name__}") from None
    metadata = head.get("Metadata") if isinstance(head.get("Metadata"), dict) else {}
    if int(head.get("ContentLength") or -1) != size or metadata.get("sha256") != digest.removeprefix("sha256:"):
        raise RemoteCpuTransportError("remote_cpu_cas_identity_mismatch")
    return True


def _publish(write_stream: Callable[[Any], None], *, digest: str, size: int, kind: str, filename: str,
             client: Any, bucket: str, failure: str) -> None:
    """Upload a repeatable stream to its CAS key and read every byte back; a changed stream is refused."""

    try:
        publish_configured_scene_stream(write_stream=write_stream, digest=digest, size_bytes=size,
                                        filename=filename, artifact_kind=kind, client=client, bucket=bucket)
    except TaskEvaluationConfiguredSceneObjectStoreError as exc:  # its message is a typed code
        raise RemoteCpuTransportError(f"{failure}:{exc}") from None


def _input_source(row: Mapping[str, Any]) -> tuple[Path, str, int]:
    """One input's host file, digest and size; the file must be a regular file of the declared size."""

    digest, size, path = row.get("digest"), row.get("size_bytes"), row.get("path")
    if (not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None or not isinstance(size, int)
            or isinstance(size, bool) or size < 0 or not isinstance(path, (str, os.PathLike))):
        raise RemoteCpuTransportError("remote_cpu_input_invalid")
    if size == 0:
        raise RemoteCpuTransportError("remote_cpu_input_empty")
    try:
        info = os.lstat(path)
    except OSError:
        raise RemoteCpuTransportError("remote_cpu_input_source_invalid") from None
    if not stat.S_ISREG(info.st_mode):
        raise RemoteCpuTransportError("remote_cpu_input_source_invalid")
    if info.st_size != size:
        raise RemoteCpuTransportError("remote_cpu_input_size_mismatch")
    return Path(path), digest, size


def _file_stream(path: Path) -> Callable[[Any], None]:
    def write(sink: Any) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise RemoteCpuTransportError("remote_cpu_input_source_invalid")
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                sink.write(chunk)

    return write


def stage_inputs(files: Sequence[Mapping[str, Any]], *, client: Any, bucket: str) -> list[dict[str, Any]]:
    """Stage each ``{"path", "digest", "size_bytes"}`` at its CAS key, once per digest.

    Every row is checked before anything is sent.  Each result is ``{"digest", "size_bytes", "uri",
    "uploaded"}``, in order; ``uploaded`` is true only for the one row that moved a digest's bytes.
    """

    rows = [_input_source(row) for row in files]
    sizes: dict[str, int] = {}
    for _, digest, size in rows:
        if sizes.setdefault(digest, size) != size:
            raise RemoteCpuTransportError("remote_cpu_input_invalid")
    staged, seen = [], set()
    for path, digest, size in rows:
        key, uploaded = cas_key(INPUT_KIND, digest, INPUT_FILENAME), False
        if digest not in seen:
            seen.add(digest)
            uploaded = not _staged(client, bucket, key, digest, size)
            if uploaded:
                _publish(_file_stream(path), digest=digest, size=size, kind=INPUT_KIND, filename=INPUT_FILENAME,
                         client=client, bucket=bucket, failure="remote_cpu_input_publication_failed")
        staged.append({"digest": digest, "size_bytes": size, "uri": f"s3://{bucket}/{key}", "uploaded": uploaded})
    return staged


def _git_command(repository: Path, *arguments: str) -> list[str]:
    # The service user reads a checkout it does not own, as the compile unit does (safe.directory).
    return ["git", "-c", f"safe.directory={repository}", "-C", str(repository), *arguments]


def _git_environment() -> dict[str, str]:
    # No optional locks: ``git status`` must not refresh (write) the index as a side effect.
    return {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


def stream_git_archive(repository: Path, source_commit: str, sink: Any, *,
                       timeout: float = ARCHIVE_TIMEOUT_SECONDS) -> None:
    """Write ``git archive --format=tar <commit> src docs/schemas pyproject.toml`` into ``sink``; git writes no file.
    Past ``timeout`` git is killed, as every other git call here is bounded."""

    process = subprocess.Popen(
        _git_command(Path(repository), "archive", "--format=tar", source_commit, *SOURCE_ARCHIVE_PATHS),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_environment())
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        process.kill()

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        with process.stdout as stdout:
            for chunk in iter(lambda: stdout.read(_CHUNK), b""):
                sink.write(chunk)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        timer.cancel()
    code = process.wait()
    if expired.is_set():
        raise RemoteCpuTransportError("remote_cpu_release_archive_timeout")
    if code != 0:
        raise RemoteCpuTransportError("remote_cpu_release_archive_failed")


class _Measured:
    """A sink that only hashes and counts."""

    def __init__(self) -> None:
        self.hash, self.size = hashlib.sha256(), 0

    def write(self, data: bytes) -> int:
        self.hash.update(data)
        self.size += len(data)
        return len(data)


def publish_release_source(*, repository: str | Path, source_commit: str, client: Any, bucket: str,
                           archive: Archive = stream_git_archive) -> dict[str, Any]:
    """Stage the release source of ``source_commit`` (recipe v2) by digest.

    The archive is streamed once to hash it; when the HEAD by digest misses it is streamed again
    into the upload, whose publisher checks the bytes against the first pass and reads them back.
    """

    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise RemoteCpuTransportError("remote_cpu_release_commit_invalid")
    root = Path(repository)
    measured = _Measured()
    archive(root, source_commit, measured)
    digest, size = "sha256:" + measured.hash.hexdigest(), measured.size
    key = cas_key(SOURCE_KIND, digest, SOURCE_FILENAME)
    uploaded = not _staged(client, bucket, key, digest, size)
    if uploaded:
        _publish(lambda sink: archive(root, source_commit, sink), digest=digest, size=size, kind=SOURCE_KIND,
                 filename=SOURCE_FILENAME, client=client, bucket=bucket,
                 failure="remote_cpu_release_source_publication_failed")
    return {"source_commit": source_commit, "recipe": SOURCE_ARCHIVE_RECIPE, "paths": list(SOURCE_ARCHIVE_PATHS),
            "digest": digest, "size_bytes": size, "uri": f"s3://{bucket}/{key}", "uploaded": uploaded}


def _git(repository: Path, *arguments: str) -> str:
    try:
        run = subprocess.run(_git_command(repository, *arguments), stdin=subprocess.DEVNULL, capture_output=True,
                             env=_git_environment(), timeout=GIT_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise RemoteCpuTransportError("remote_cpu_release_git_unavailable") from None
    if run.returncode != 0:
        raise RemoteCpuTransportError("remote_cpu_release_git_failed")
    return run.stdout.decode("utf-8", "replace").strip()


def release_commit(repository: str | Path) -> str:
    """The checkout's HEAD commit, clean as the compile unit requires (``status --untracked-files=no``)."""

    root = Path(repository).resolve()
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    if _COMMIT.fullmatch(commit) is None:
        raise RemoteCpuTransportError("remote_cpu_release_commit_invalid")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RemoteCpuTransportError("remote_cpu_release_checkout_dirty")
    return commit


def publish_running_release(object_store: tuple[Any, str, str]) -> dict[str, Any]:
    """Stage the release this code runs from: the clean HEAD of its own checkout (the allocator's preflight)."""

    repository = Path(__file__).resolve().parents[2]
    return publish_release_source(repository=repository, source_commit=release_commit(repository),
                                  client=object_store[0], bucket=object_store[1])
