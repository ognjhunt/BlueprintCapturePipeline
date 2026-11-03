"""The paid remote team-scene unit's ExecCondition, with the standard library alone (plan 14 §1).

The condition runs every 60 s, so it reads the filesystem instead of importing the collector: the unit runs
when one of this stage's leases is live (``live/rcj-tp-*``), a hand-off or shadow marker of this stage waits,
or the effective mode is a remote one, in that order, so nothing about the mode can keep a drain from its run.
A set ``BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION`` keeps its name, and an invalid one runs as ``host``. Unset or
empty is auto: remote once the remote-CPU config is there and names ``team_scene_packet``, as the paid unit
would load it (``task_evaluation_episode_compilation_remote_condition.configured`` checks file, mode, size,
JSON, schema, seal, shape, stage and US region), and ``host`` without it, which skips and reads nothing but
the config's path. Its constants are pinned to the remote module's and the contract's by
``tests/test_team_scene_packet_remote_units.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .task_evaluation_episode_compilation_remote_condition import (
    CONFIG_ENV,
    DEFAULT_CONFIG_PATH,
    DEFAULT_JOBS_ROOT,
    JOBS_ROOT_ENV,
    _waiting,
    configured,
)

EXECUTION_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION"
REMOTE_MODES = ("cloud_run_shadow", "cloud_run")
STAGE = "team_scene_packet"
# Every job id of this stage starts so (``remote_cpu_job_contract.job_id_for``): ``rcj-<abbreviation>-``.
JOB_PREFIX = "rcj-tp-"
MARKER_DIRECTORIES = ("handoffs", "shadow")


def remote_mode(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the effective mode is a remote one: a set remote mode, or an unset one beside this stage's config."""

    values = os.environ if environ is None else environ
    requested = str(values.get(EXECUTION_ENV) or "").strip()
    if requested:
        return requested in REMOTE_MODES
    return configured(values.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH, STAGE)


def should_run(jobs_root: str | Path, environ: Mapping[str, str] | None = None) -> bool:
    root = Path(jobs_root)
    return (_waiting(root / "live", JOB_PREFIX) or any(_waiting(root / name / STAGE) for name in MARKER_DIRECTORIES)
            or remote_mode(environ))


def main(argv: list[str] | None = None) -> int:
    """Exit 0 to run the unit, 1 to skip it; an optional argument names the jobs root."""

    arguments = sys.argv[1:] if argv is None else argv
    jobs_root = arguments[0] if arguments else (os.environ.get(JOBS_ROOT_ENV) or DEFAULT_JOBS_ROOT)
    return 0 if should_run(jobs_root) else 1


__all__ = ["main", "remote_mode", "should_run"]


if __name__ == "__main__":
    raise SystemExit(main())
