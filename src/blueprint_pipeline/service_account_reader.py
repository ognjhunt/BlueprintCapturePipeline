"""Read published bytes back as the service account through one process.

A deploy proves that the unprivileged service account can read every byte of
each runtime tree it publishes (the splat renderer with its Chromium and
``node_modules``, the scene-configuration toolchain, the website runtime). The
proof used to start ``sudo -u <account> dd`` once per file -- thousands of
privilege transitions per deploy. This reader starts that one account process
once and serves every path through it. The proof is unchanged: the service
account opens and reads every byte, and the caller (root) still checks the
length and digest of exactly the bytes the account read.

Bytes stream in bounded chunks. Reading each file whole held every byte twice
on the root side and once in the account process; on 2026-10-13 the first
ARTISAN toolchain (libtorch_cuda.so is 1.19 GB) failed the deploy that way.
``digest`` hashes the stream without keeping it.

``identity`` has the account open a file for reading and report its own
descriptor's sealed identity without reading the bytes: for an inode an earlier
release's readback already read in full (see ``readback_ledger``).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

# Runs as the service account under ``python -I -S``: no site, no environment.
# Request: operation byte (r: read, i: identity) + 4-byte big-endian length +
# path bytes. Reply: status byte. On 0 (read) the bytes follow as chunks, each a
# 4-byte big-endian length + bytes, ended by a zero-length chunk; on 0
# (identity) an 8-byte big-endian length + the opened descriptor's sealed
# identity as space-separated decimals. On 1 (refused) an 8-byte big-endian
# length + the refusal follows. A read error after the first chunk ends the
# stream with length 0xFFFFFFFF, then the refusal as on 1.
CHUNK_BYTES = 4 * 1024 * 1024
_READER = r"""
import os, sys
requests, replies = sys.stdin.buffer, sys.stdout.buffer
while True:
    head = requests.read(5)
    if len(head) < 5:
        break
    operation, path = head[:1], requests.read(int.from_bytes(head[1:], "big"))
    try:
        stream = open(path, "rb")
    except OSError as exc:
        body = repr(exc).encode("utf-8", "replace")
        replies.write(b"\1" + len(body).to_bytes(8, "big") + body)
        replies.flush()
        continue
    if operation == b"i":
        with stream:
            value = os.fstat(stream.fileno())
        body = " ".join(str(field) for field in (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_mode, value.st_uid, value.st_gid)).encode()
        replies.write(b"\0" + len(body).to_bytes(8, "big") + body)
        replies.flush()
        continue
    replies.write(b"\0")
    try:
        with stream:
            while True:
                chunk = stream.read(%d)
                if not chunk:
                    break
                replies.write(len(chunk).to_bytes(4, "big") + chunk)
        replies.write((0).to_bytes(4, "big"))
    except OSError as exc:
        body = repr(exc).encode("utf-8", "replace")
        replies.write(b"\xff\xff\xff\xff" + len(body).to_bytes(8, "big") + body)
    replies.flush()
""" % CHUNK_BYTES
_FAILED_STREAM = 0xFFFFFFFF


def _read_exactly(stream: IO[bytes], count: int) -> bytes:
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise EOFError("service_account_reader_exited")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


class ServiceAccountReader:
    """A ``Readback`` callable backed by one long-lived service-account process."""

    def __init__(
        self,
        user: str,
        *,
        python: str = sys.executable,
        command_prefix: Sequence[str] | None = None,
    ) -> None:
        if not user.strip():
            raise ValueError("service_account_reader_user_required")
        argv = (
            self.command(user, python=python)
            if command_prefix is None
            else [*command_prefix, python, "-I", "-S", "-c", _READER]
        )
        self._argv = argv
        self._process = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.closed = False

    @staticmethod
    def command(user: str, *, python: str = sys.executable) -> list[str]:
        return ["sudo", "-n", "-u", user, "--", python, "-I", "-S", "-c", _READER]

    def __call__(self, path: Path) -> bytes:
        """Every byte the service account read; use ``digest`` for large files."""
        chunks: list[bytes] = []
        self._stream(path, chunks.append)
        return b"".join(chunks)

    def digest(self, path: Path) -> tuple[int, str]:
        """(length, ``sha256:<hex>``) of the bytes the account read, held one chunk at a time."""
        hasher, size = hashlib.sha256(), 0

        def consume(chunk: bytes) -> None:
            nonlocal size
            hasher.update(chunk)
            size += len(chunk)

        self._stream(path, consume)
        return size, "sha256:" + hasher.hexdigest()

    def identity(self, path: Path) -> tuple[int, ...]:
        """``sealed_identity`` of the inode the account opened for reading at ``path``; no bytes read."""
        fields: list[int] = []
        self._stream(path, fields.extend, operation=b"i")
        return tuple(fields)

    def _stream(self, path: Path, consume, *, operation: bytes = b"r") -> None:
        if self.closed:
            raise ValueError("service_account_reader_closed")
        encoded = os.fsencode(Path(path))
        process = self._process
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(operation + len(encoded).to_bytes(4, "big") + encoded)
            process.stdin.flush()
            status = _read_exactly(process.stdout, 1)
            if status != b"\0":
                refusal = _read_exactly(process.stdout, int.from_bytes(_read_exactly(process.stdout, 8), "big"))
                # The per-file reader raised CalledProcessError when dd refused.
                raise subprocess.CalledProcessError(1, ["read", str(path)], stderr=refusal)
            if operation == b"i":
                body = _read_exactly(process.stdout, int.from_bytes(_read_exactly(process.stdout, 8), "big"))
                consume(int(field) for field in body.split())
                return
            while True:
                length = int.from_bytes(_read_exactly(process.stdout, 4), "big")
                if length == 0:
                    return
                if length == _FAILED_STREAM:
                    refusal = _read_exactly(process.stdout, int.from_bytes(_read_exactly(process.stdout, 8), "big"))
                    raise subprocess.CalledProcessError(1, ["read", str(path)], stderr=refusal)
                consume(_read_exactly(process.stdout, length))
        except (BrokenPipeError, EOFError) as exc:
            # Surface why the account process ended (a refused sudo, a missing
            # interpreter), as the per-file reader's dd error did.
            reason = self._finish()
            raise subprocess.CalledProcessError(
                process.returncode if process.returncode is not None else 1,
                self._argv[:5],
                stderr=reason or b"service_account_reader_exited",
            ) from exc

    def _finish(self) -> bytes:
        """Stop the account process; return what it wrote to stderr (bounded)."""
        if self.closed:
            return b""
        self.closed = True
        process = self._process
        reason = b""
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            if process.stderr is not None:
                try:
                    reason = process.stderr.read(64 * 1024)
                except OSError:
                    reason = b""
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        return reason

    def close(self) -> None:
        self._finish()

    def __enter__(self) -> "ServiceAccountReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
