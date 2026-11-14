"""The host's side of the off-host GPU reaper: a heartbeat and every live Vast watchdog's deadline.

Every two minutes (``blueprint-off-host-reaper-heartbeat.timer``) this reads the command line of
every running ``groot_oscar_runpod_watchdog --provider vast`` process, which is each Vast
machine's hard deadline as the host holds it now (a superseded watchdog's successor carries the
new one), and rewrites one object in the reaper bucket (``off_host_gpu_reaper.STATE_OBJECT``):
the heartbeat time, each watchdog's label scope, instance id when its launch recorded one, and
deadline, and the longest deadline this release can arm. While the host is up the reaper sees
current deadlines; when it is not, the last published ones stand.

Renting never waits on this. When publishing has failed for longer than
``PUBLISH_FAILURE_PAGE_SECONDS`` it pages once, on the capacity controller's webhook, and is
quiet again until a publish succeeds and fails anew. With no bucket configured it does nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.common import write_json
from .off_host_gpu_reaper import (
    BUCKET_ENV,
    LABEL_PATTERN,
    STATE_OBJECT,
    STATE_SCHEMA_VERSION,
    GcsObjectStore,
    ObjectStore,
    parse_host_state,
)

REPORT_ROOT_ENV = "BLUEPRINT_OFF_HOST_GPU_REAPER_HEARTBEAT_ROOT"
DEFAULT_REPORT_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/off-host-gpu-reaper")
CREDENTIAL_FILE_ENV = "BLUEPRINT_OFF_HOST_GPU_REAPER_CREDENTIAL_FILE"
DEFAULT_CREDENTIAL_FILE = "/etc/blueprint/credentials/off-host-gpu-reaper-heartbeat.json"
STATUS_NAME = "heartbeat.json"
STATUS_SCHEMA_VERSION = "off_host_gpu_reaper_heartbeat_status.v1"
PUBLISH_FAILURE_PAGE_SECONDS = 10 * 60
PAGE_CODE = "off_host_reaper_heartbeat_failing"
WATCHDOG_MODULE = "blueprint_pipeline.groot_oscar_runpod_watchdog"
#: ``vast_independent_watchdog_control.write_started_vast_instance_id`` writes it in the out dir.
STARTED_INSTANCE_ID_NAME = "started_vast_instance_id.txt"
#: The hard-TTL ceilings of the lanes that rent Vast machines. The longest is the bound for a
#: machine the host published no deadline for.
DEADLINE_CEILINGS = (
    ("task_evaluation_scene_configuration_paid_authority", "MAX_TTL_SECONDS"),
    ("public_scene_artifixer3d_vast", "MAX_TTL_SECONDS"),
    ("public_scene_aura_exact_residual_vast", "MAX_TTL_SECONDS"),
    ("team_candidate_quick10_admission", "MAX_TTL_SECONDS"),
    ("native_task_arena_paid_authority", "MAX_TTL_SECONDS"),
    ("reconstruction_gpu_admission", "MAX_TTL_SECONDS"),
    ("openpi_policy_ranking_gpu_admission", "MAX_TTL_SECONDS"),
    ("deployment_embodiment_preparation_contract", "MAX_TTL_SECONDS"),
    ("canonical_3dgs_admission", "MAX_TTL_SECONDS"),
    ("beta_policy_screening_capabilities", "DEFAULT_HARD_TTL_SECONDS"),
    ("sam31_gpu_admission", "MAX_TTL_SECONDS"),
    ("paired_target_native_import_vast", "MAX_TTL_SECONDS"),
    ("rigid_robot_free_probe_lane", "HARD_TTL_SECONDS"),
    ("semantic_teacher_image_edit_paid_authority", "MAX_TTL_SECONDS"),
)


class ScanIncomplete(RuntimeError):
    """A process the scan could not read, or a watchdog whose arguments it could not parse."""


def longest_code_deadline_seconds() -> float:
    import importlib

    values = [float(getattr(importlib.import_module(f"blueprint_pipeline.{module}"), name))
              for module, name in DEADLINE_CEILINGS]
    return max(values)


def process_argvs(proc_root: Path = Path("/proc")) -> tuple[list[list[str]], list[str]]:
    """Every running process's argv, and why the scan is incomplete (empty when it read every process).

    A process that exits mid-scan is skipped: it holds no deadline. One that cannot be read is a problem.
    """

    argvs: list[list[str]] = []
    problems: list[str] = []
    # A directory lists in no fixed order; read processes by number so the result does not depend on it.
    for entry in sorted((item for item in proc_root.iterdir() if item.name.isdigit()), key=lambda item: int(item.name)):
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            problems.append(f"process_unreadable:{type(error).__name__}")
            continue
        if raw:
            argvs.append([part.decode("utf-8", errors="replace") for part in raw.rstrip(b"\0").split(b"\0")])
    return argvs, problems


def _options(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            name, equals, value = token.partition("=")
            if not equals:
                if index + 1 >= len(tokens):
                    raise ScanIncomplete("watchdog_arguments_unparseable")
                value, index = tokens[index + 1], index + 1
            if name in {"--out-dir", "--pod-name-prefix", "--deadline-epoch", "--provider", "--resource-name-exact"}:
                values[name] = value
        index += 1
    return values


def _started_instance_id(out_dir: str) -> int | None:
    try:
        text = (Path(out_dir) / STARTED_INSTANCE_ID_NAME).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() and int(text) > 0 else None


def watchdog_entries(argvs: Iterable[Sequence[str]], *,
                     started_instance_id: Callable[[str], int | None] = _started_instance_id,
                     ) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per live Vast watchdog (label scope, machine id when recorded, deadline), and the
    watchdogs it could not read. A list with problems is never presented as whole."""

    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for argv in argvs:
        try:
            entry = _watchdog_entry(argv, started_instance_id)
        except ScanIncomplete as error:
            problems.append(str(error))
            continue
        if entry is not None:
            entries.append(entry)
    # One order whatever order the processes were listed in: the published object must not change
    # between two heartbeats that saw the same watchdogs.
    entries.sort(key=lambda row: (row["deadline_epoch"], str(row.get("label") or row.get("label_prefix") or ""),
                                  row.get("instance_id") or 0))
    return entries, sorted(problems)


def _watchdog_entry(argv: Sequence[str], started_instance_id: Callable[[str], int | None]) -> dict[str, Any] | None:
    try:
        module_at = next(index for index in range(len(argv) - 1)
                         if argv[index] == "-m" and argv[index + 1] == WATCHDOG_MODULE)
    except StopIteration:
        return None
    options = _options(argv[module_at + 2:])
    if options.get("--provider", "runpod") != "vast":
        return None
    try:
        deadline = float(options["--deadline-epoch"])
    except (KeyError, ValueError):
        raise ScanIncomplete("watchdog_deadline_unparseable") from None
    prefix, exact = options.get("--pod-name-prefix", ""), options.get("--resource-name-exact")
    if not math.isfinite(deadline) or deadline <= 0 or not LABEL_PATTERN.fullmatch(prefix) or (
            exact is not None and not LABEL_PATTERN.fullmatch(exact)):
        raise ScanIncomplete("watchdog_scope_unparseable")
    entry: dict[str, Any] = {"deadline_epoch": deadline, "instance_id": None}
    if exact:
        entry["label"] = exact
    else:
        entry["label_prefix"] = prefix
    if options.get("--out-dir"):
        entry["instance_id"] = started_instance_id(options["--out-dir"])
    return entry


def build_host_state(entries: Sequence[Mapping[str, Any]], *, now: float, longest: float,
                     complete: bool) -> dict[str, Any]:
    state = {"schema_version": STATE_SCHEMA_VERSION, "heartbeat_epoch": now,
             "published_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
             "longest_deadline_seconds": longest, "deadlines_complete": bool(complete),
             "entries": [dict(entry) for entry in entries]}
    parse_host_state(json.dumps(state).encode("utf-8"))  # the reaper must be able to read what is published
    return state


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def publish(*, store_factory: Callable[[], ObjectStore] | None, report_root: Path, now: float,
            argvs: Callable[[], tuple[list[list[str]], list[str]]] = process_argvs,
            longest: Callable[[], float] = longest_code_deadline_seconds, webhook_url: str = "",
            poster: Callable[[str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """One heartbeat. Returns the status record it wrote; never raises for a publish failure."""

    path = report_root / STATUS_NAME
    previous = _read_status(path)
    record: dict[str, Any] = {"schema_version": STATUS_SCHEMA_VERSION, "last_attempt_epoch": now,
                              "last_success_epoch": previous.get("last_success_epoch"),
                              "failing_since_epoch": None, "paged_epoch": None, "last_error": None}
    if store_factory is None:
        record["status"] = "not_configured"
        write_json(path, record)
        return record
    reason: str | None = None
    try:
        found, scan_problems = argvs()
        entries, parse_problems = watchdog_entries(found)
        problems = [*scan_problems, *parse_problems]
        state = build_host_state(entries, now=now, longest=longest(), complete=not problems)
        store_factory().write(STATE_OBJECT, json.dumps(state, sort_keys=True).encode("utf-8"))
    except Exception as error:  # noqa: BLE001 - every failure is recorded and, past the window, paged
        reason = type(error).__name__
    else:
        record.update(last_success_epoch=now, published_deadline_count=len(state["entries"]),
                      deadlines_complete=not problems)
        if problems:
            # Published, so the heartbeat stays fresh, but marked incomplete; it pages like a failure.
            reason = "deadlines_incomplete:" + problems[0]
    if reason is not None:
        since = previous.get("failing_since_epoch")
        since = float(since) if isinstance(since, (int, float)) and not isinstance(since, bool) else now
        record.update(status="failing" if record.get("deadlines_complete") is None else "published_incomplete",
                      last_error=reason, failing_since_epoch=since,
                      paged_epoch=previous.get("paged_epoch"), page_error=None)
        if record["paged_epoch"] is None and now - since > PUBLISH_FAILURE_PAGE_SECONDS:
            alert = {"code": PAGE_CODE, "severity": "page", "provider": "off_host_gpu_reaper",
                     "reason": reason, "failing_seconds": round(now - since)}
            try:
                if not webhook_url:
                    raise RuntimeError("operator_alert_route_unconfigured")
                from . import control_plane_capacity_controller as capacity

                (poster or capacity.post_alert)(webhook_url, {"level": "critical", "mounts": [], "alerts": [alert]})
            except Exception as page_error:  # noqa: BLE001 - retried next tick; it may carry the URL
                record["page_error"] = str(page_error) if isinstance(page_error, RuntimeError) and str(
                    page_error) == "operator_alert_route_unconfigured" else type(page_error).__name__
            else:
                record["paged_epoch"] = now
    else:
        record["status"] = "published"
    write_json(path, record)
    return record


def _store_factory(bucket: str, credential_file: str) -> Callable[[], ObjectStore]:
    def build() -> ObjectStore:
        from .cloud_run_jobs_client import load_service_account_file

        credentials = load_service_account_file(credential_file, error_prefix="off_host_gpu_reaper_heartbeat")
        return GcsObjectStore(bucket, credentials=credentials, project=getattr(credentials, "project_id", None))

    return build


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the host's heartbeat and Vast deadlines for the off-host reaper.")
    parser.add_argument("--report-root", default=os.getenv(REPORT_ROOT_ENV) or str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args(argv)
    bucket = str(os.getenv(BUCKET_ENV) or "").strip()
    credential_file = str(os.getenv(CREDENTIAL_FILE_ENV) or DEFAULT_CREDENTIAL_FILE)
    from .control_plane_capacity_controller import WEBHOOK_URL_ENV

    record = publish(store_factory=_store_factory(bucket, credential_file) if bucket else None,
                     report_root=Path(args.report_root), now=time.time(),
                     webhook_url=str(os.getenv(WEBHOOK_URL_ENV) or "").strip())
    print(json.dumps(record, sort_keys=True))
    return 0


__all__ = [
    "DEADLINE_CEILINGS", "PAGE_CODE", "PUBLISH_FAILURE_PAGE_SECONDS", "ScanIncomplete", "build_host_state",
    "longest_code_deadline_seconds", "process_argvs", "publish", "watchdog_entries",
]


if __name__ == "__main__":
    raise SystemExit(main())
