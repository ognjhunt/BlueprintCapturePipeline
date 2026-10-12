"""Read published bytes back as the service account through one process.

A deploy proves that the unprivileged service account can read every byte of
each runtime tree it publishes (the splat renderer with its Chromium and
``node_modules``, the scene-configuration toolchain, the website runtime). The
proof used to start ``sudo -u <account> dd`` once per file -- thousands of
privilege transitions per deploy. This reader starts that one account process
once and serves every path through it. The proof is unchanged: the service
account opens and reads every byte, and the caller (root) still checks the
length and digest of exactly the bytes the account read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

# Runs as the service account under ``python -I -S``: no site, no environment.
# Request: 4-byte big-endian length + path bytes. Reply: status byte (0 read,
# 1 refused) + 8-byte big-endian length + file bytes or the refusal.
_READER = r"""
import sys
requests, replies = sys.stdin.buffer, sys.stdout.buffer
while True:
    head = requests.read(4)
    if len(head) < 4:
        break
    path = requests.read(int.from_bytes(head, "big"))
    try:
        with open(path, "rb") as stream:
            body, status = stream.read(), b"\0"
    except OSError as exc:
        body, status = repr(exc).encode("utf-8", "replace"), b"\1"
    replies.write(status + len(body).to_bytes(8, "big") + body)
    replies.flush()
"""


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
        if self.closed:
            raise ValueError("service_account_reader_closed")
        encoded = os.fsencode(Path(path))
        process = self._process
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(len(encoded).to_bytes(4, "big") + encoded)
            process.stdin.flush()
            status = _read_exactly(process.stdout, 1)
            body = _read_exactly(process.stdout, int.from_bytes(_read_exactly(process.stdout, 8), "big"))
        except (BrokenPipeError, EOFError) as exc:
            # Surface why the account process ended (a refused sudo, a missing
            # interpreter), as the per-file reader's dd error did.
            reason = self._finish()
            raise subprocess.CalledProcessError(
                process.returncode if process.returncode is not None else 1,
                self._argv[:5],
                stderr=reason or b"service_account_reader_exited",
            ) from exc
        if status != b"\0":
            # The per-file reader raised CalledProcessError when dd refused.
            raise subprocess.CalledProcessError(1, ["read", str(path)], stderr=body)
        return body

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
