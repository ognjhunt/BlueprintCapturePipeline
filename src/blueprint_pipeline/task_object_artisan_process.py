"""Bound the official controller and its descendants to an absolute deadline."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


def _descendants(parent):
    children, identities = {}, {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
            identities[int(entry.name)] = fields[19]
        except (OSError, ValueError, IndexError):
            pass
    result, queue = {}, [parent]
    while queue:
        current = queue.pop()
        for child in children.get(current, []):
            result[child] = identities[child]
            queue.append(child)
    return result


def _start_time(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]
    except (OSError, IndexError):
        return None


def run_controller(argv, *, cwd, env, timeout, check=False, capture_output=True, text=True):
    if not capture_output or not text:
        raise ValueError("artisan_controller_output_protocol_invalid")
    deadline = time.monotonic() + float(timeout)
    if float(timeout) <= 0:
        raise subprocess.TimeoutExpired(argv, timeout)
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    identity = _start_time(process.pid)
    descendants = {}
    try:
        while True:
            descendants.update(_descendants(process.pid))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        descendants.update(_descendants(process.pid))
        for pid, started in {**descendants, process.pid: identity}.items():
            if started is None or _start_time(pid) != started:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if identity is not None and _start_time(process.pid) == identity:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
