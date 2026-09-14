"""Bound scene component subprocesses together with their background children."""
from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - callers bind the executable and inputs


def run_component_process(command, *, timeout, capture_output=True, check=False, **kwargs):
    """Match the component runner interface and close its process group on failure."""
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:  # nosec B603
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.stdout, exc.stderr = stdout, stderr
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
