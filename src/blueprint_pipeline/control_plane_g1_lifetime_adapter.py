"""Dedicated bounded handshake for the controlled optional G1 worker protocol."""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any

from .control_plane_lane_scratch import LaneScratchError
from .control_plane_scratch_lifetime import LeasedScratchUse, _path

MAX_MESSAGE_BYTES = 4096
HANDSHAKE_SECONDS = 5.0


def _wire_size(value: Any, depth: int = 0) -> int:
    if depth > 12:
        raise LaneScratchError("lane_scratch_handshake_invalid")
    if isinstance(value, str):
        if len(value) > MAX_MESSAGE_BYTES:
            raise LaneScratchError("lane_scratch_handshake_invalid")
        size = 2
        for char in value:
            if 0xD800 <= ord(char) <= 0xDFFF:
                raise LaneScratchError("lane_scratch_handshake_invalid")
            size += 2 if char in ('"', "\\") else 6 if ord(char) < 32 else len(char.encode("utf-8"))
        return size
    if type(value) is int and value.bit_length() <= 64:
        return len(str(value))
    if isinstance(value, (list, tuple)) and len(value) <= 16:
        return 2 + max(0, len(value) - 1) + sum(_wire_size(v, depth + 1) for v in value)
    if isinstance(value, dict) and len(value) <= 16:
        return 2 + max(0, len(value) - 1) + sum(_wire_size(k, depth + 1) + 1 + _wire_size(v, depth + 1) for k, v in value.items())
    raise LaneScratchError("lane_scratch_handshake_invalid")


def write_message(fd: int, value: dict[str, Any]) -> None:
    if _wire_size(value) + 1 > MAX_MESSAGE_BYTES:
        raise LaneScratchError("lane_scratch_handshake_invalid")
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    try:
        if os.write(fd, raw) != len(raw):
            raise LaneScratchError("lane_scratch_handshake_invalid")
    except OSError:
        raise LaneScratchError("lane_scratch_handshake_invalid") from None


def read_message(fd: int, *, timeout: float = HANDSHAKE_SECONDS,
                 _deadline: float | None = None) -> dict[str, Any]:
    if type(timeout) not in (int, float) or not 0 < timeout <= HANDSHAKE_SECONDS or not math.isfinite(timeout):
        raise LaneScratchError("lane_scratch_handshake_invalid")
    started = time.monotonic()
    if _deadline is not None:
        try:
            valid = (type(_deadline) in (int, float) and math.isfinite(_deadline)
                     and _deadline <= started + timeout)
        except OverflowError:
            valid = False
        if not valid:
            raise LaneScratchError('lane_scratch_handshake_invalid')
    deadline = started + timeout if _deadline is None else _deadline
    raw = bytearray()
    try:
        while b"\n" not in raw:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise LaneScratchError("lane_scratch_handshake_timeout")
            chunk = os.read(fd, min(512, MAX_MESSAGE_BYTES + 1 - len(raw)))
            if not chunk or len(raw) + len(chunk) > MAX_MESSAGE_BYTES:
                raise LaneScratchError("lane_scratch_handshake_invalid")
            raw.extend(chunk)
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise LaneScratchError("lane_scratch_handshake_invalid")
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError
                value[key] = item
            return value
        value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict) or not value or _wire_size(value) + 1 > MAX_MESSAGE_BYTES:
            raise LaneScratchError("lane_scratch_handshake_invalid")
        return value
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise LaneScratchError("lane_scratch_handshake_invalid") from None


def worker_proof(use: LeasedScratchUse, *, output: Path, request_digest: str) -> dict[str, Any]:
    with use.borrow(output):
        pass
    proof = {"identity": {**use.identity, "inodes": list(use.identity["inodes"])},
             "output": str(output), "request_digest": request_digest}
    if _wire_size(proof) + 1 > MAX_MESSAGE_BYTES:
        raise LaneScratchError("lane_scratch_handshake_invalid")
    return proof


def adopt_worker_proof(fd: int, proof: dict[str, Any], *, output: Path, request_digest: str,
                       now: Any = time.time, _owner: LeasedScratchUse | None = None) -> LeasedScratchUse:
    owner = _owner if _owner is not None else LeasedScratchUse()
    try:
        if fd not in owner._owned:
            owner._take(fd)
        if (not isinstance(proof, dict) or set(proof) != {"identity", "output", "request_digest"}
                or proof["output"] != str(_path(output)) or proof["request_digest"] != request_digest):
            raise LaneScratchError("lane_scratch_handshake_invalid")
        original = owner._detach(fd)
        use = LeasedScratchUse.inherited(fd, proof["identity"], now=now, _owned_identity=original)
        try:
            with use.borrow(output):
                pass
            return use
        except BaseException:
            use.close()
            raise
    except OSError:
        raise LaneScratchError("lane_scratch_handshake_invalid") from None
    finally:
        if _owner is None:
            owner.close()


def controlled_worker_run(*, executable: Path, request: Path, output: Path, request_digest: str,
                          use: LeasedScratchUse, stdout: Any, timeout: float) -> subprocess.CompletedProcess:
    """Keep parent SH while the acknowledged direct child owns its inherited SH."""
    proof = worker_proof(use, output=output, request_digest=request_digest)
    owner = LeasedScratchUse()
    process = None
    try:
        to_child, parent_write = os.pipe()
        owner._take_all((to_child, parent_write))
        parent_read, from_child = os.pipe()
        owner._take_all((parent_read, from_child))
        process = subprocess.Popen([str(executable), "-m", "blueprint_pipeline.native_g1_development_worker",
                                    "--request", str(request), "--output-dir", str(output),
                                    "--lifetime-fd", str(use.fd), "--lifetime-input-fd", str(to_child),
                                    "--lifetime-output-fd", str(from_child)], stdout=stdout, stderr=subprocess.STDOUT,
                                   pass_fds=(use.fd, to_child, from_child))
        owner._close_one(to_child)
        owner._close_one(from_child)
        if to_child in owner._owned or from_child in owner._owned:
            raise LaneScratchError("lane_scratch_descriptor_cleanup_failed")
        owner._cleanup_status()
        write_message(parent_write, proof)
        if read_message(parent_read) != {"status": "ready", "request_digest": request_digest}:
            raise LaneScratchError("lane_scratch_handshake_invalid")
        write_message(parent_write, {"status": "proceed"})
        return subprocess.CompletedProcess(process.args, process.wait(timeout=timeout))
    except BaseException as error:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=HANDSHAKE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                raise LaneScratchError("lane_scratch_handshake_child_finalization_failed") from None
        if isinstance(error, OSError):
            raise LaneScratchError("lane_scratch_handshake_invalid") from None
        raise
    finally:
        owner.close()
