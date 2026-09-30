"""The paid remote episode-compilation unit's ExecCondition, with the standard library alone (plan 14 §1).

The condition runs every 60 s.  Importing the collector to answer it costs about 0.8 s of CPU and
75 MB each time, so this reads the filesystem instead: the unit runs when a remote mode is requested,
a lease is live, or a hand-off or shadow marker waits.  In ``host`` mode with nothing outstanding it
skips.  An invalid mode runs as ``host``.  The collector's own effective mode (which also waits for
the owner census) decides what a run then does.  Its constants are pinned to the remote module's by
``tests/test_task_evaluation_episode_compilation_units.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

EXECUTION_ENV = "BLUEPRINT_EPISODE_COMPILATION_EXECUTION"
REMOTE_MODES = ("cloud_run_shadow", "cloud_run")
JOBS_ROOT_ENV = "BLUEPRINT_REMOTE_CPU_JOBS_ROOT"
DEFAULT_JOBS_ROOT = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
STAGE = "episode_compilation"
MARKER_DIRECTORIES = ("handoffs", "shadow")


def _waiting(directory: Path) -> bool:
    """Whether a directory holds a record; a dot file is one still being written."""

    try:
        with os.scandir(directory) as entries:
            return any(not entry.name.startswith(".") for entry in entries)
    except OSError:
        return False


def should_run(jobs_root: str | Path, environ: Mapping[str, str] | None = None) -> bool:
    requested = str((os.environ if environ is None else environ).get(EXECUTION_ENV) or "").strip()
    root = Path(jobs_root)
    return (requested in REMOTE_MODES or _waiting(root / "live")
            or any(_waiting(root / name / STAGE) for name in MARKER_DIRECTORIES))


def main(argv: list[str] | None = None) -> int:
    """Exit 0 to run the unit, 1 to skip it; an optional argument names the jobs root."""

    arguments = sys.argv[1:] if argv is None else argv
    jobs_root = arguments[0] if arguments else (os.environ.get(JOBS_ROOT_ENV) or DEFAULT_JOBS_ROOT)
    return 0 if should_run(jobs_root) else 1


__all__ = ["main", "should_run"]


if __name__ == "__main__":
    raise SystemExit(main())
