"""Replay a SAM child's saved job against the code in this tree, paying for nothing.

The slow loop of 2026-09-05 was: deploy a fix, resubmit, wait fifteen minutes for
the chain to reach the stage that failed, learn one more fact, repeat.  Three
deploys were spent on one review bug (frame set, authority linkage, wrapper
paths) that the retained inputs of the failed child could have exposed in one
local run.  This command is that run, as a first-class step: it locates the
child's saved job in the phase queue, validates it exactly as the worker does,
executes the stage handler from *this* tree against the retained inputs in a
fresh scratch root, and reports the outcome with the predicate that refused
(``fail_closed_blocker_explainer``).  The queue is never written.

Rules the command enforces:

- Paid phases and the hardware calibration render are not replayed unless
  ``--allow-paid`` is passed; a replay is for the CPU and review boundaries.
- ``--isolate`` re-executes under ``systemd-run`` as the service user with
  ``PrivateNetwork=yes``, so a review replay stops at the model call and a
  paid lane cannot reach a provider even by mistake.
- The server profile the job was executed with is discovered by the digest
  the plan pins, so a replay never needs the current release's profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import task_evaluation_sam31_preparation_execution as execution
from . import task_evaluation_sam31_preparation_stages as stages
from . import task_evaluation_scene_configuration_sam31_preparation_driver as driver
from .task_evaluation_launch_preparation_queue import QUEUE_STATES as PARENT_QUEUE_STATES
from .fail_closed_blocker_explainer import explain_blocker, fired_predicates
from .task_evaluation_scene_configuration_sam31_plan import PROFILE_ENV

SCHEMA = "task_evaluation_stage_replay_report.v1"
JOB_STATES = ("failed", "completed", "waiting_external", "processing", "pending")
DEFAULT_QUEUE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions")
DEFAULT_PARENT_QUEUE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations")
DEFAULT_INPUT_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/prepared-references")
DEFAULT_REPLAY_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/stage-replays")
DEFAULT_APPROVED_ROOTS = (Path("/var/lib/blueprint"), Path("/opt/blueprint"), Path("/etc/blueprint"))
DEFAULT_ENVIRONMENT_FILES = ("/etc/blueprint/pipeline-control-plane.env",)
_BOUNDARY_MARKERS = ("connection", "timeout", "unreachable", "name resolution", "gaierror", "urlerror", "apiconnection",
                     "openaiofficialcostgate", "provider_zero", "vast_api")


@dataclass(frozen=True)
class LocatedChild:
    job_path: Path
    result_path: Path
    state: str


def locate_child(queue_root: str | Path, child_id: str) -> LocatedChild:
    """Find the saved job of ``child_id`` in whichever queue state holds it."""

    root = Path(queue_root)
    for state in JOB_STATES:
        candidate = root / state / f"{child_id}.json"
        if candidate.is_file():
            return LocatedChild(job_path=candidate, result_path=root / "results" / f"{child_id}.json", state=state)
    raise FileNotFoundError(f"{child_id} is not in any state of {root}")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def discover_input_root(job: Mapping[str, Any]) -> Path | None:
    """The preparation input root the job was materialized under.

    The plan is written at ``<input_root>/<preparation_id>/<digest>`` and the
    content-addressed store the worker validates against lives beside it, so the
    job itself says where to look; a replay must not guess a root the worker never
    used (the first host run did, one level too high, and every blob was "missing").
    """

    plan_path = Path(str((job.get("plan_ref") or {}).get("path") or ""))
    for candidate in plan_path.parents[1:3] if plan_path.is_absolute() else ():
        if (candidate / "content-addressed" / "sha256").is_dir():
            return candidate
    return None


def discover_server_profile(plan_path: Path, input_root: Path) -> Path | None:
    """The profile the job ran under: the one whose digest the plan pins."""

    try:
        pinned = str(_read(plan_path).get("server_profile_sha256") or "")
    except (OSError, ValueError):
        return None
    if not pinned:
        return None
    current = os.environ.get(PROFILE_ENV)
    candidates = [Path(current)] if current else []
    for root in (Path(input_root), Path(input_root).parent):
        candidates.extend(sorted(root.glob("*/sam31-hardware-profile*/sam31_preparation_profile.v1.json")))
    for candidate in candidates:
        try:
            if candidate.is_file() and _sha(candidate) == pinned:
                return candidate
        except OSError:
            continue
    return None


def _blocker(exc: BaseException) -> str:
    if isinstance(exc, (execution.Sam31PhaseExecutionError, ValueError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"[:300]


def _boundary_hint(exc: BaseException) -> str | None:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _BOUNDARY_MARKERS):
        return "external_boundary_unreachable_by_design"
    return None


def _explained(report: dict[str, Any], status: str, exc: BaseException) -> dict[str, Any]:
    report.update(
        status=status,
        blocker=_blocker(exc),
        exception_type=type(exc).__name__,
        fired_predicates=fired_predicates(exc),
        explanation=explain_blocker(exc),
        boundary_hint=_boundary_hint(exc),
    )
    return report


def _write_report(run_root: Path, report: Mapping[str, Any]) -> str:
    path = run_root / "stage_replay_report.v1.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path)


def replay_child(
    *,
    queue_root: str | Path,
    child_id: str,
    parent_queue_root: str | Path,
    input_root: str | Path,
    replay_root: str | Path,
    approved_roots: Sequence[str | Path] = DEFAULT_APPROVED_ROOTS,
    allow_paid: bool = False,
) -> dict[str, Any]:
    """Run the saved job of ``child_id`` through this tree's stage handler in a scratch root."""

    located = locate_child(queue_root, child_id)
    job = _read(located.job_path)
    saved = _read(located.result_path) if located.result_path.is_file() else None
    if not (Path(input_root) / "content-addressed" / "sha256").is_dir():
        input_root = discover_input_root(job) or input_root
    Path(replay_root).mkdir(parents=True, exist_ok=True)
    run_root = Path(
        tempfile.mkdtemp(prefix=f"{child_id[:20]}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-", dir=str(replay_root))
    )
    phase = str(job.get("phase") or "")
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "child_id": child_id,
        "phase": phase,
        "queue_state": located.state,
        "job_path": str(located.job_path),
        "saved_result": None
        if saved is None
        else {"status": saved.get("status"), "blocker": saved.get("blocker"), "source_commit": saved.get("source_commit")},
        "replay_root": str(run_root),
        "code_root": str(Path(stages.__file__).resolve().parents[2]),
        "server_profile": os.environ.get(PROFILE_ENV),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fired_predicates": [],
        "paid_execution_requested": False,
        "provider_mutation_performed": False,
    }
    paid = phase in stages.PAID_PHASES or phase == "calibrated_views"
    if paid and not allow_paid:
        report.update(status="paid_stage_not_replayed", blocker="replay_refuses_paid_phase_without_allow_paid")
        report["report_path"] = _write_report(run_root, report)
        return report
    try:
        request, plan = execution._validated_job(
            job,
            parent_queue=Path(parent_queue_root),
            input_root=Path(input_root),
            source_commit=str(job.get("expected_source_commit") or ""),
            approved_roots=tuple(Path(root) for root in approved_roots),
        )
    except Exception as exc:  # noqa: BLE001 - the refusal is the finding
        _explained(report, "job_refused", exc)
        report["report_path"] = _write_report(run_root, report)
        return report
    context = {
        **job,
        "request": request,
        "plan": plan,
        "queue_root": str(queue_root),
        "output_root": str(run_root),
        "preparation_input_root": str(input_root),
        "resume_only": False,
        "previous_progress": None,
    }
    try:
        outcome = stages.execute_stage(context)
    except Exception as exc:  # noqa: BLE001 - the refusal is the finding
        _explained(report, "refused", exc)
    else:
        status = str(outcome.get("status") or "")
        report["outcome"] = outcome
        if status == "completed":
            report["status"] = "completed"
        elif status == "waiting_for_external_result":
            report["status"] = "waiting"
        else:
            report["status"] = "refused"
            blockers = outcome.get("blockers") or []
            report["blocker"] = ";".join(str(item) for item in blockers)[:700] or "stage_failed"
            lowered = report["blocker"].lower()
            if any(marker in lowered for marker in _BOUNDARY_MARKERS):
                report["boundary_hint"] = "external_boundary_unreachable_by_design"
    report["report_path"] = _write_report(run_root, report)
    return report


# --------------------------------------------------------------------------- #
# Parent-level replay: the whole parent worker pass on a scratch queue
# --------------------------------------------------------------------------- #

PARENT_SCHEMA = "task_evaluation_parent_replay_report.v1"
ALLOWED_URI_PREFIXES_ENV = "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_ALLOWED_URI_PREFIXES_JSON"


class ReplayBoundary(RuntimeError):
    """Raised by the replay's stand-ins where production would render, fetch or pay."""


@dataclass(frozen=True)
class LocatedParent:
    envelope_path: Path
    state: str
    stem: str


def locate_parent(parent_queue_root: str | Path, preparation_id: str) -> LocatedParent:
    """Find the parent envelope by exact stem or by a prefix that matches exactly one file."""

    root = Path(parent_queue_root)
    matches: list[tuple[str, Path]] = []
    for state in PARENT_QUEUE_STATES:
        directory = root / state
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.stem == preparation_id or path.stem.startswith(preparation_id):
                matches.append((state, path))
    if not matches:
        raise FileNotFoundError(f"{preparation_id} is not in any state of {root}")
    if len(matches) > 1:
        raise FileNotFoundError(f"{preparation_id} matches {len(matches)} envelopes; give the full stem")
    state, path = matches[0]
    return LocatedParent(envelope_path=path, state=state, stem=path.stem)


def envelope_uri_prefixes(envelope: Mapping[str, Any]) -> list[str]:
    """The URI prefixes (scheme://authority/first-segment/) the request's references use."""

    prefixes: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            uri = value.get("uri")
            if isinstance(uri, str) and "://" in uri:
                scheme, _, rest = uri.partition("://")
                parts = rest.split("/")
                prefixes.add(f"{scheme}://{parts[0]}/" + (f"{parts[1]}/" if len(parts) > 2 else ""))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(envelope.get("request") or {})
    return sorted(prefixes)


def _refusing_fetcher(calls: list[str]):
    def fetch(uri: str, destination: Path, size: int) -> None:
        calls.append(str(uri))
        raise ReplayBoundary("reference_fetch_not_replayed")
    return fetch


def _render_boundary(**_kwargs: Any) -> dict[str, Any]:
    raise ReplayBoundary("scene_render_inputs_not_replayed")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_children_of(parent_digest: str, child_queue_root: Path, scratch_child: Path) -> int:
    copied = 0
    for state in ("results", "completed", "failed", "waiting_external"):
        directory = child_queue_root / state
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            try:
                value = _read(path)
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and value.get("parent_request_digest") == parent_digest:
                _link_or_copy(path, scratch_child / state / path.name)
                copied += 1
    return copied


def replay_parent(
    *,
    parent_queue_root: str | Path,
    preparation_id: str,
    child_queue_root: str | Path,
    input_root: str | Path,
    replay_root: str | Path,
    allowed_uri_prefixes: Sequence[str],
    service_account: str | None = None,
    advancer: Any | None = None,
) -> dict[str, Any]:
    """Re-run the parent worker for one preparation on a scratch queue and report the row it produces.

    The envelope goes into the scratch queue's ``pending``; its progress and resume
    records are copied so the driver continues the same sequence; the children's
    results are copied so the driver reads them where the queue expects; the
    content store is reused through hard links so nothing is fetched; disk
    reservations, pins and downstream queues point at scratch.  Rendering and
    fetching raise ``ReplayBoundary``, so reaching the render step means the SAM
    chain was accepted as ``ready``.  Production queues are never written.
    """

    from . import task_evaluation_launch_preparation_worker as worker

    located = locate_parent(parent_queue_root, preparation_id)
    envelope = _read(located.envelope_path)
    request = envelope.get("request") or {}
    parent_digest = str(envelope.get("request_digest") or "")
    Path(replay_root).mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f"parent-{located.stem[:24]}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-", dir=str(replay_root)))
    scratch_queue = run_root / "launch-preparations"
    scratch_child = run_root / "sam31-preparation-executions"
    scratch_inputs = run_root / "prepared-references"
    for name in ("pending",):
        (scratch_queue / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(located.envelope_path, scratch_queue / "pending" / located.envelope_path.name)
    source_queue = Path(parent_queue_root)
    for progress_dir in ("source-progress", "source-resume-completed", "source-resume-pending", "source-resume-blocked"):
        origin = source_queue / progress_dir / located.stem
        if origin.is_dir():
            shutil.copytree(origin, scratch_queue / progress_dir / located.stem)
    children_copied = _copy_children_of(parent_digest, Path(child_queue_root), scratch_child)
    real_store = Path(input_root) / "content-addressed" / "sha256"
    linked = 0
    if real_store.is_dir():
        for blob in real_store.iterdir():
            if blob.is_file():
                _link_or_copy(blob, scratch_inputs / "content-addressed" / "sha256" / blob.name)
                linked += 1
    scratch_inputs.mkdir(parents=True, exist_ok=True)
    fetch_calls: list[str] = []
    render_reached = False
    def render_boundary(**kwargs: Any) -> dict[str, Any]:
        nonlocal render_reached
        render_reached = True
        return _render_boundary(**kwargs)
    account = service_account or pwd.getpwuid(os.geteuid()).pw_name
    previous_child_env = os.environ.get(driver.CHILD_QUEUE_ENV)
    os.environ[driver.CHILD_QUEUE_ENV] = str(scratch_child)
    report: dict[str, Any] = {
        "schema_version": PARENT_SCHEMA,
        "preparation_id": request.get("preparation_id"),
        "envelope_path": str(located.envelope_path),
        "queue_state": located.state,
        "request_digest": parent_digest,
        "source_commit": request.get("expected_production_commit"),
        "scratch_queue_root": str(scratch_queue),
        "scratch_child_queue_root": str(scratch_child),
        "children_copied": children_copied,
        "content_blobs_linked": linked,
        "server_profile": os.environ.get(PROFILE_ENV),
        "code_root": str(Path(stages.__file__).resolve().parents[2]),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paid_execution_requested": False,
        "provider_mutation_performed": False,
    }
    try:
        outcome = worker.process_launch_preparation_queue(
            queue_root=scratch_queue, input_root=scratch_inputs, allowed_uri_prefixes=list(allowed_uri_prefixes),
            service_account=account, source_commit=str(request.get("expected_production_commit") or ""),
            max_messages=1, fetcher=_refusing_fetcher(fetch_calls),
            sam31_preparation_advancer=advancer, scene_render_input_materializer=render_boundary,
            construction_queue_root=run_root / "scene-constructions",
            episode_compilation_queue_root=run_root / "episode-compilations",
            # No disk reservation: a replay reads production evidence and writes only scratch; the
            # admission floor belongs to the production worker, not to a diagnosis on any host.
            disk_reservation_root=None, storage_pins_root=run_root / "storage-pins",
        )
    except Exception as exc:  # noqa: BLE001 - the refusal is the finding
        _explained(report, "worker_refused", exc)
        report["report_path"] = _write_report(run_root, report)
        return report
    finally:
        if previous_child_env is None:
            os.environ.pop(driver.CHILD_QUEUE_ENV, None)
        else:
            os.environ[driver.CHILD_QUEUE_ENV] = previous_child_env
    rows = outcome.get("results") or []
    row = rows[0] if rows else {}
    blockers = [str(item) for item in (row.get("blockers") or [])]
    boundary = render_reached and not fetch_calls and any("ReplayBoundary" in item for item in blockers)
    report.update(
        status=str(row.get("status") or "no_row"),
        row={key: row.get(key) for key in ("status", "blockers", "preparation_id", "observed_at_iso")},
        advancement=row.get("advancement"),
        fired_predicates=[item.split(":predicates=", 1)[1] for item in blockers if ":predicates=" in item],
        nothing_fetched=not fetch_calls,
        fetch_attempts=fetch_calls[:8],
        reached_render_inputs_boundary=boundary,
        sam31_ready=boundary,
    )
    result_path = scratch_queue / "results" / located.envelope_path.name
    admission = replay_next_consumers(result_path=result_path, queue_root=scratch_queue) if result_path.is_file() else []
    report.update(
        next_consumer_admission=admission,
        next_consumers_admitted=bool(admission) and all(row["status"] == "accepted" for row in admission),
    )
    report["report_path"] = _write_report(run_root, report)
    return report


def replay_next_consumers(*, result_path: Path, queue_root: Path) -> list[dict[str, Any]]:
    """Replay the admission predicates of the workers that read a preparation result next.

    A parent that reaches ``queued_for_production_scene_configuration`` is next read by the
    activation automation and by the controls-intent provisioner; each re-validates the result
    and its intake envelope with predicates of its own.  On 2026-09-06 the activation automation
    refused the real 841757 envelope over a schema name no producer writes, after the whole paid
    SAM chain had completed.  Replaying the consumers' own validators here names that in a
    second, for nothing.
    """

    from . import fail_closed_blocker_explainer as explainer
    from . import task_evaluation_configured_controls_continuation_provisioning as controls
    from . import task_evaluation_scene_configuration_activation_automation as activation

    commit = str(_read(result_path).get("source_commit") or "")
    consumers = (
        ("task_evaluation_scene_configuration_activation_automation", activation._preparation_context,
         {"preparation_result_path": result_path, "preparation_queue_root": queue_root}),
        ("task_evaluation_configured_controls_continuation_provisioning", controls._preparation_context,
         {"preparation_result_path": result_path, "preparation_queue_root": queue_root,
          "expected_production_commit": commit}),
    )
    rows: list[dict[str, Any]] = []
    for name, validator, kwargs in consumers:
        outcome = explainer.explain_call(validator, **kwargs)
        row: dict[str, Any] = {"consumer": name, "status": outcome["status"]}
        if outcome["status"] == "refused":
            row["blocker"] = outcome["blocker"]
            row["fired_predicates"] = [p for e in outcome["explanations"] for p in e["fired"]]
        rows.append(row)
    return rows


def isolation_command(
    argv: Sequence[str],
    *,
    user: str = "blueprint",
    environment_files: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    working_directory: str | None = None,
) -> list[str]:
    """Wrap ``argv`` so it runs as the service user with no network at all."""

    command = [
        "systemd-run", "--wait", "--pipe", "--collect", "--quiet", "--service-type=exec",
        f"--unit=stage-replay-{os.getpid()}", "-p", "PrivateNetwork=yes", "-p", f"User={user}",
        "-p", "TimeoutStartSec=1800",
        # The caller's working directory is the code root of the replay: a candidate
        # checkout's ``src`` resolves the package before any PYTHONPATH an
        # EnvironmentFile may override (files win over --setenv in systemd).
        "-p", f"WorkingDirectory={working_directory or os.getcwd()}",
    ]
    for path in environment_files:
        command.extend(["-p", f"EnvironmentFile={path}"])
    for key, value in sorted((environment or {}).items()):
        command.append(f"--setenv={key}={value}")
    command.append("--")
    command.extend(argv)
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a SAM child's saved job against this tree's code without paying.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--child", help="child id (sam31-<digest>)")
    target.add_argument("--parent", help="parent preparation envelope stem or unique prefix (replays the parent worker pass)")
    parser.add_argument("--allowed-uri-prefix", action="append", default=[], help=f"repeatable; defaults to the JSON list in {ALLOWED_URI_PREFIXES_ENV}")
    parser.add_argument("--service-account", default=None, help="parent replay: the account the worker must run as (default: current user)")
    parser.add_argument("--queue-root", default=str(DEFAULT_QUEUE_ROOT))
    parser.add_argument("--parent-queue-root", default=str(DEFAULT_PARENT_QUEUE_ROOT))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    parser.add_argument("--approved-root", action="append", default=[], help="repeatable; defaults to the production roots")
    parser.add_argument("--server-profile", default=None, help="profile the job ran under; discovered from the plan digest when omitted")
    parser.add_argument("--allow-paid", action="store_true", help="replay a paid phase (needs the provider environment; never default)")
    parser.add_argument("--isolate", action="store_true", help="re-run under systemd-run as --user with PrivateNetwork=yes")
    parser.add_argument("--user", default="blueprint")
    parser.add_argument("--environment-file", action="append", default=list(DEFAULT_ENVIRONMENT_FILES))
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.isolate:
        inner = [sys.executable, "-m", "blueprint_pipeline.task_evaluation_stage_replay",
                 *(["--child", args.child] if args.child else ["--parent", args.parent]),
                 "--queue-root", args.queue_root, "--parent-queue-root", args.parent_queue_root,
                 "--input-root", args.input_root, "--replay-root", args.replay_root]
        for prefix in args.allowed_uri_prefix:
            inner.extend(["--allowed-uri-prefix", prefix])
        if args.service_account:
            inner.extend(["--service-account", args.service_account])
        for root in args.approved_root:
            inner.extend(["--approved-root", root])
        if args.server_profile:
            inner.extend(["--server-profile", args.server_profile])
        if args.allow_paid:
            inner.append("--allow-paid")
        if args.json_out:
            inner.extend(["--json-out", args.json_out])
        environment = {"PYTHONPATH": os.environ.get("PYTHONPATH", "")} if os.environ.get("PYTHONPATH") else {}
        completed = subprocess.run(
            isolation_command(inner, user=args.user, environment_files=args.environment_file, environment=environment),
            check=False,
        )
        return completed.returncode
    if args.parent:
        prefixes = list(args.allowed_uri_prefix)
        if not prefixes and os.environ.get(ALLOWED_URI_PREFIXES_ENV):
            try:
                prefixes = [str(item) for item in json.loads(os.environ[ALLOWED_URI_PREFIXES_ENV])]
            except ValueError:
                prefixes = []
        if not prefixes:
            # A replay fetches nothing, so the prefixes only have to admit the URIs the
            # envelope already carries; derive them from the envelope itself rather than
            # depend on how the unit's environment reaches an isolated shell.
            prefixes = envelope_uri_prefixes(_read(locate_parent(args.parent_queue_root, args.parent).envelope_path))
        if args.server_profile:
            os.environ[PROFILE_ENV] = args.server_profile
        elif not os.environ.get(PROFILE_ENV):
            located_parent = locate_parent(args.parent_queue_root, args.parent)
            digest = str(_read(located_parent.envelope_path).get("request_digest") or "")
            for path in sorted(Path(args.queue_root).glob("*/*.json")):
                try:
                    job = _read(path)
                except (OSError, ValueError):
                    continue
                if isinstance(job, dict) and job.get("parent_request_digest") == digest and job.get("plan_ref"):
                    discovered = discover_server_profile(Path(str(job["plan_ref"].get("path") or "")), Path(args.input_root))
                    if discovered is not None:
                        os.environ[PROFILE_ENV] = str(discovered)
                        break
        report = replay_parent(
            parent_queue_root=args.parent_queue_root, preparation_id=args.parent, child_queue_root=args.queue_root,
            input_root=args.input_root, replay_root=args.replay_root, allowed_uri_prefixes=prefixes,
            service_account=args.service_account,
        )
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json_out:
            Path(args.json_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        admitted = report.get("reached_render_inputs_boundary") or report.get("next_consumers_admitted")
        return 0 if admitted else (3 if str(report.get("status")).startswith("waiting") else 2)
    if args.server_profile:
        os.environ[PROFILE_ENV] = args.server_profile
    else:
        try:
            located = locate_child(args.queue_root, args.child)
            plan_path = Path(str(_read(located.job_path).get("plan_ref", {}).get("path") or ""))
        except (OSError, ValueError, AttributeError):
            plan_path = Path("")
        discovered = discover_server_profile(plan_path, Path(args.input_root)) if plan_path.is_file() else None
        if discovered is not None:
            os.environ[PROFILE_ENV] = str(discovered)
    report = replay_child(
        queue_root=args.queue_root, child_id=args.child, parent_queue_root=args.parent_queue_root,
        input_root=args.input_root, replay_root=args.replay_root,
        approved_roots=tuple(args.approved_root) or DEFAULT_APPROVED_ROOTS, allow_paid=args.allow_paid,
    )
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return {"completed": 0, "waiting": 3}.get(str(report.get("status")), 2)


__all__ = ["LocatedChild", "LocatedParent", "ReplayBoundary", "discover_input_root", "discover_server_profile", "envelope_uri_prefixes", "locate_parent", "replay_next_consumers", "replay_parent", "isolation_command", "locate_child", "main", "replay_child"]

if __name__ == "__main__":
    raise SystemExit(main())
