"""The paid remote episode-compilation unit's ExecCondition, with the standard library alone (plan 14 §1).

The condition runs every 60 s.  Importing the collector to answer it costs about 0.8 s of CPU and
75 MB each time, so this reads the filesystem instead: the unit runs when a lease is live, a hand-off
or shadow marker waits, or the effective mode is a remote one, in that order, so nothing about the
mode can keep a drain from its run.  A set mode keeps its name, and an invalid one runs as ``host``.
An unset or empty mode is auto (owner decision 2026-09-30): remote once this stage's remote-CPU
config is there, as the paid unit would load it, and ``host`` without it, which skips exactly as
before and reads nothing but the config's path.

``configured`` checks what the standard library can cheaply: a regular, unlinked file no other
account may write or read, no larger than the allocator reads, holding a sealed
``remote_cpu_workers_config.v1`` object of the config's own shape, in a US region, that names this
stage.  Anything it cannot read is no config.  A sealed config that fails a deeper check (an image
not pinned by digest, a malformed rate table) still starts the unit, whose run loads it as the
allocator does, finds it unusable and drains without a provider connection.  The collector's own
effective mode (which also waits for the owner census) decides what a run then does.  Its constants
are pinned to the remote module's, the contract's and the allocator's by
``tests/test_task_evaluation_episode_compilation_units.py`` and
``tests/test_task_evaluation_episode_compilation_remote_condition.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

EXECUTION_ENV = "BLUEPRINT_EPISODE_COMPILATION_EXECUTION"
REMOTE_MODES = ("cloud_run_shadow", "cloud_run")
JOBS_ROOT_ENV = "BLUEPRINT_REMOTE_CPU_JOBS_ROOT"
DEFAULT_JOBS_ROOT = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
CONFIG_ENV = "BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/blueprint/remote-cpu-workers.json"
CONFIG_SCHEMA_VERSION = "remote_cpu_workers_config.v1"
CONFIG_MAX_BYTES = 256 * 1024
# The config's shape and region rule, as ``remote_cpu_job_contract.config_blockers`` checks them.
CONFIG_KEYS = frozenset({"schema_version", "project", "region", "transport_bucket", "stages", "rate_table",
                         "max_live_executions", "max_attempts", "config_digest"})
STAGE_KEYS = frozenset({"job", "image", "vcpu", "memory_bytes", "ephemeral_bytes", "task_timeout_seconds"})
US_REGION = re.compile(r"us-[a-z]+[0-9]+")
# Group write and any access by other accounts: the allocator refuses such a config (0640 at most).
CONFIG_FORBIDDEN_MODE = 0o027
STAGE = "episode_compilation"
MARKER_DIRECTORIES = ("handoffs", "shadow")


def _waiting(directory: Path) -> bool:
    """Whether a directory holds a record; a dot file is one still being written."""

    try:
        with os.scandir(directory) as entries:
            return any(not entry.name.startswith(".") for entry in entries)
    except OSError:
        return False


def configured(path: str | Path) -> bool:
    """Whether the remote-CPU config at ``path`` is there for this stage, as far as the standard library can tell."""

    try:
        return _configured(path)
    except Exception:  # noqa: BLE001 - review M1: deep nesting, a lone surrogate: what cannot be read is no config
        return False


def _configured(path: str | Path) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        status = os.fstat(descriptor)  # before any stream wraps it: wrapping a directory raises
        if not stat.S_ISREG(status.st_mode) or status.st_mode & CONFIG_FORBIDDEN_MODE:
            return False
        payload = b""
        while len(payload) <= CONFIG_MAX_BYTES:  # one byte past the bound is enough to refuse it
            chunk = os.read(descriptor, CONFIG_MAX_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload += chunk
    except OSError:  # a config that cannot be read now is none: skipped, and asked again in a minute
        return False
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload) if len(payload) <= CONFIG_MAX_BYTES else None
    except ValueError:
        return False
    if (not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA_VERSION or set(value) != CONFIG_KEYS
            or not isinstance(value["region"], str) or US_REGION.fullmatch(value["region"]) is None):
        return False
    stages = value["stages"]
    if not isinstance(stages, dict) or not isinstance(stages.get(STAGE), dict) or set(stages[STAGE]) != STAGE_KEYS:
        return False
    # The config's own seal, as ``decision_evidence_contracts.canonical_digest`` computes it.
    sealed = json.dumps({name: item for name, item in value.items() if name != "config_digest"}, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False)
    return value.get("config_digest") == "sha256:" + hashlib.sha256(sealed.encode("utf-8")).hexdigest()


def remote_mode(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the effective mode is a remote one: a set remote mode, or an unset one beside this stage's config."""

    values = os.environ if environ is None else environ
    requested = str(values.get(EXECUTION_ENV) or "").strip()
    if requested:
        return requested in REMOTE_MODES
    return configured(values.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)


def should_run(jobs_root: str | Path, environ: Mapping[str, str] | None = None) -> bool:
    root = Path(jobs_root)
    return (_waiting(root / "live") or any(_waiting(root / name / STAGE) for name in MARKER_DIRECTORIES)
            or remote_mode(environ))


def main(argv: list[str] | None = None) -> int:
    """Exit 0 to run the unit, 1 to skip it; an optional argument names the jobs root."""

    arguments = sys.argv[1:] if argv is None else argv
    jobs_root = arguments[0] if arguments else (os.environ.get(JOBS_ROOT_ENV) or DEFAULT_JOBS_ROOT)
    return 0 if should_run(jobs_root) else 1


__all__ = ["configured", "main", "remote_mode", "should_run"]


if __name__ == "__main__":
    raise SystemExit(main())
