"""Replay an Atlas preview job's next controller steps against this tree, paying for nothing.

Deploying a fix and waiting about an hour for the job's next tick taught one fact
per deploy. This runs ``website_atlas_preview.advance_job`` from this tree on a
scratch copy of a pulled capture root until the job stops, and reports where it
stopped and what it would buy there. Retained receipts are adopted by the real
code paths; only a new purchase, an effect nothing local can answer, or a network
call is stopped. Pull the capture root to its own absolute path, as the stage
replay does; a missing input comes back as ``missing_input`` naming the path to pull.
A failed or stopped job is first retried in the copy, as the operator would after
the deploy. ``first_stop.kind`` is ``would_buy``, ``refusal``, ``missing_input``,
``external_effect`` or ``error``. A retained record that was not pulled reads as
absent and can turn into a purchase or refusal, so a stop lists ``unpulled_before``:
what the code looked for just before it and did not find.

The pulled evidence is never written. Where ``unshare`` is permitted, the copy is
bind-mounted over the capture root's own absolute path inside a private mount and
network namespace, so retained records naming absolute host paths resolve to the
copy. Elsewhere it runs at ``<run>/mirror/<absolute capture root>``
(``overlay.mode`` says which). The evidence digest is compared before and after.

Boundaries; no flag lifts any of them:

- WebApp control: ``task-context`` answers the retained context and
  ``atlas-configuration`` the snapshot (``--configuration``, else the job's retained
  ``configuration.json``). ``preparation-spend`` is admitted under the snapshot's
  dollar ceiling exactly as WebApp admits it (money is the only stop; request
  counts are recorded), then stops as ``would_buy``. Every other operation stops as
  ``external_effect``.
- Any other paid admission grant stops as ``would_buy``. The GCP monthly report is
  host-only and is not evaluated.
- Release blockers admit this tree's HEAD: the replay stands in for its deploy.
- Provider credentials are placeholders that never leave the process.
- World Labs transport and archive publication stop as ``external_effect``;
  WebApp delivery and provider cleanup are skipped.
- Sockets and network tools fail as ``unstubbed_network_call:<host or tool>``.
"""
from __future__ import annotations

import argparse
import builtins
import errno
import hashlib
import io
import json
import math
import os
import pathlib
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .common import write_json
from .website_atlas_preview_store import ARTIFACT_KIND, JOB_FILE, LOCK_FILE, TERMINAL, read_job

SCHEMA = "atlas_preview_replay_report.v1"
MAXIMUM_STEPS = 32
PLACEHOLDER = "replay-placeholder-never-sent"
DEFAULT_ATLAS_BASE = "https://api.atlas-beta.worldlabs.ai/api/v2"
_NETWORK_TOOLS = frozenset({"curl", "wget", "gsutil", "gcloud", "ssh", "scp", "sftp", "rsync", "aws", "b2", "nc",
                            "ncat", "telnet"})
_GIT_NETWORK = frozenset({"fetch", "pull", "push", "clone", "ls-remote", "remote", "submodule"})
_PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_DEAD_PROXY = "http://127.0.0.1:9"
_LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "::1"})
#: Keys only a paid call's own record carries: an admission, a provider receipt or a settlement.
_PAID_RECORD_KEYS = frozenset({"admission", "allocation_binding_digest", "operation_receipt", "planner_receipt",
                               "settlement", "provider_charge_amount_usd", "cost_usd", "request_digest"})
_STUB_MODULES = frozenset({__name__, "blueprint_pipeline.website_task_context",
                           "blueprint_pipeline.paid_resource_admission"})
_TICK_FIELDS = frozenset({"last_tick_epoch", "updated_at_epoch", "next_attempt_at_epoch"})
_MISSING = object()


class ReplayStop(BaseException):
    """Raised by a stub; a BaseException, so no ``except Exception`` in the controller absorbs it."""

    def __init__(self, kind: str, code: str, **detail: Any) -> None:
        super().__init__(f"{kind}:{code}")
        self.kind, self.code, self.detail = kind, code, detail

    def report(self) -> dict[str, Any]:
        return {"kind": self.kind, "stage": self.detail.get("stage"), "code": self.code,
                **{key: value for key, value in self.detail.items() if key != "stage"}}


def _package_modules() -> list[Any]:
    return [module for name, module in list(sys.modules.items())
            if module is not None and (name == "blueprint_pipeline" or name.startswith("blueprint_pipeline."))]


def _swap(stack: ExitStack, module: Any, name: str, value: Any) -> None:
    """Replace ``module.name`` and every loaded alias of it; restore each one on exit."""
    original = getattr(module, name)
    for loaded in (module, *_package_modules()):
        if vars(loaded).get(name) is original:
            setattr(loaded, name, value)

    def restore() -> None:
        for loaded in (module, *_package_modules()):
            if vars(loaded).get(name) is value:
                setattr(loaded, name, original)
    stack.callback(restore)


def _set(stack: ExitStack, owner: Any, name: str, value: Any) -> None:
    own = vars(owner).get(name, _MISSING)
    setattr(owner, name, value)
    stack.callback(lambda: delattr(owner, name) if own is _MISSING else setattr(owner, name, own))


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for directory, names, files in os.walk(root):
        names.sort()
        for name in sorted([*files, *(name for name in names if os.path.islink(os.path.join(directory, name)))]):
            path = Path(directory, name)
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update((os.readlink(path) if path.is_symlink() else _file_digest(path)).encode() + b"\0")
    return "sha256:" + digest.hexdigest()


def _source_state(root: Path) -> tuple[str, bool]:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return (head.stdout.strip() if head.returncode == 0 else ""), status.returncode == 0 and not status.stdout.strip()


def _atlas_base(root: Path) -> str:
    """The API root the job's own submissions used, else the configured or documented one (not a secret)."""
    from .website_atlas_pose import ATLAS_API_BASE_URL_ENV
    used = set()
    for path in root.rglob("submission.json"):
        try:
            base = _read_json(path).get("api_base")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(base, str):
            used.add(base)
    configured = (os.environ.get(ATLAS_API_BASE_URL_ENV) or os.environ.get("WLT_API_BASE_URL") or "").strip()
    return used.pop() if len(used) == 1 else configured.rstrip("/") or DEFAULT_ATLAS_BASE


def _raised_at(error: BaseException) -> str | None:
    """The innermost three package frames that raised, outermost first."""
    sites, trace = [], error.__traceback__
    while trace is not None:
        module = trace.tb_frame.f_globals.get("__name__", "")
        if module.startswith("blueprint_pipeline.") and module != __name__:
            sites.append(f"{module.removeprefix('blueprint_pipeline.')}.{trace.tb_frame.f_code.co_name}:{trace.tb_lineno}")
        trace = trace.tb_next
    return " > ".join(sites[-3:]) or None


class _Harness:
    """Every stub of one replay, and what each saw."""

    def __init__(self, *, capture_root: Path, workspace: Path, prefix: str | None, configuration: Any,
                 task_context: Any, commit: str) -> None:
        self.capture_root, self.workspace, self.prefix, self.commit = capture_root, workspace, prefix, commit
        # The scene directory: a withdrawal tombstone lives beside the captures.
        self.watched = str(workspace.parents[1])
        self.configuration, self.task_context = configuration, task_context
        self.stage, self.stages, self.would_buy, self.refused_spend = None, [], [], []
        self.stops, self.network, self.skipped, self.failure, self.failure_stage = [], [], [], None, None
        self.reads, self.absent, self.adopted = {}, {}, {}
        pulled = [(path.relative_to(workspace).as_posix(), path.is_dir()) for path in workspace.rglob("*")
                  if not path.is_symlink()]
        self.retained = {relative for relative, directory in pulled if not directory}
        self.pulled_dirs = {"", *(relative for relative, directory in pulled if directory)}
        self.atlas_base = _atlas_base(workspace)
        self._open = io.open

    def host(self, value: Any) -> str:
        text = os.fsdecode(value)
        if self.prefix and (text == self.prefix or text.startswith(self.prefix + os.sep)):
            return text[len(self.prefix):] or "/"
        return text

    def host_text(self, text: str) -> str:
        return text.replace(self.prefix, "") if self.prefix else text

    def unpulled(self) -> list[str]:
        """Paths the code looked for that still do not exist, except in directories the replay itself made."""
        workspace, rows = str(self.workspace) + os.sep, []
        for path in self.absent:
            local = (self.prefix or "") + path
            parent = os.path.dirname(local)
            made = (parent.startswith(workspace) and parent[len(workspace):] not in self.pulled_dirs
                    and os.path.isdir(parent))
            if not os.path.lexists(local) and not made:
                rows.append(path)
        return rows

    def stop(self, kind: str, code: str, **detail: Any) -> None:
        stop = ReplayStop(kind, code, **{"stage": self.stage, **detail})
        self.stops.append(stop)
        raise stop

    def reached(self, stage: Any) -> None:
        if isinstance(stage, str) and stage:
            self.stage = stage
            if not self.stages or self.stages[-1] != stage:
                self.stages.append(stage)

    def snapshot(self, name: str) -> dict[str, Any]:
        value = getattr(self, name)
        if isinstance(value, str):
            self.stop("missing_input", value)
        return value

    def origin(self) -> dict[str, Any]:
        task = site = None
        frame = sys._getframe(1)
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("blueprint_pipeline.") and module not in _STUB_MODULES:
                site = site or f"{module.removeprefix('blueprint_pipeline.')}.{frame.f_code.co_name}:{frame.f_lineno}"
                value = frame.f_locals.get("task")
                task = task or (value if isinstance(value, str) and value.isidentifier() else None)
            frame = frame.f_back
        return {"stage": self.stage, "task": task, "call_site": site}

    def webapp(self, *, capture_id: str, operation: str, payload: Mapping[str, Any],
               source_kind: str = "website") -> dict[str, Any]:
        if operation == "task-context":
            return dict(self.snapshot("task_context"))
        if operation == "atlas-configuration":
            from .website_atlas_pipeline import REVIEW_CAPS
            value = {key: item for key, item in self.snapshot("configuration").items() if key != REVIEW_CAPS}
            return {**value, "grant_creation_allowed": False} if payload.get("create") is False else value
        if operation == "preparation-spend" and payload.get("create") is not False:
            return self.admit(dict(payload["spend"]))
        self.stop("external_effect", "webapp:" + operation)

    def admit(self, spend: dict[str, Any]) -> dict[str, Any]:
        """WebApp's reservation: an Atlas-lane call over the allowance refuses; any other is a purchase."""
        configuration = self.snapshot("configuration")
        held = {row.get("allocation_binding_digest") for row in
                ((configuration.get("allowance_accounting") or {}).get("reservations") or [])}
        provider, cost = spend.get("provider"), spend.get("maximum_cost_usd")
        row = {"boundary": "webapp:preparation-spend",
               **{key: spend.get(key) for key in ("resource_class", "provider", "maximum_cost_usd", "request_count",
                                                  "allocation_binding_digest")},
               "already_reserved": spend.get("allocation_binding_digest") in held, **self.origin(),
               "unpulled_before": self.unpulled()[-10:]}
        atlas = (provider == "world_labs_atlas" or spend.get("atlas_auxiliary") is True
                 or provider == "openai" and bool(spend.get("source_video_digest")))
        ceiling = (configuration.get("authority") or {}).get("maximum_cost_usd")
        reserved = configuration.get("reserved_or_settled_cost_usd")
        remaining = configuration.get("remaining_cost_usd")
        numbers = all(type(value) in (int, float) for value in (cost, ceiling, reserved))
        if atlas and not row["already_reserved"] and type(cost) in (int, float):
            # websiteSceneSponsorship.ts: reserved + micros(cost) > floor(allowance * 1e6) refuses.
            if numbers:
                over = math.ceil(reserved * 1_000_000) + math.ceil(cost * 1_000_000) > math.floor(ceiling * 1_000_000)
            else:
                over = type(remaining) in (int, float) and math.ceil(cost * 1_000_000) > math.floor(remaining * 1_000_000)
            row["allowance"] = {"maximum_cost_usd": ceiling, "reserved_or_settled_cost_usd": reserved,
                                "remaining_cost_usd": remaining}
            if over:
                self.refused_spend.append(row)
                raise ValueError("website_control_preparation-spend_http_409:website_scene_preparation_budget_exhausted")
        self.would_buy.append(row)
        self.stop("would_buy", ":".join(str(part) for part in (provider, row["task"]) if part), **{
            key: row[key] for key in ("boundary", "resource_class", "provider", "task", "maximum_cost_usd",
                                      "request_count", "call_site")})

    def grant(self, admission: Mapping[str, Any], *, resource_class: str, expected_schema_version: str) -> Any:
        row = {"boundary": "paid_resource_admission", "resource_class": resource_class,
               **{key: admission.get(key) for key in ("provider", "maximum_cost_usd", "request_count",
                                                      "allocation_binding_digest")},
               "already_reserved": False, **self.origin(), "unpulled_before": self.unpulled()[-10:]}
        self.would_buy.append(row)
        self.stop("would_buy", ":".join(str(part) for part in (row["provider"] or resource_class, row["task"]) if part),
                  boundary=row["boundary"], resource_class=resource_class, call_site=row["call_site"])

    def transport(self, _base: str) -> Any:
        def send(method: str, url: str, _headers: Any, _data: Any) -> Any:
            self.stop("external_effect", f"atlas_provider:{method} {urlsplit(url).path}")
        return send

    def skip(self, name: str) -> Any:
        def skipped(job: dict[str, Any], **_options: Any) -> dict[str, Any]:
            self.skipped.append(name)
            return job
        return skipped

    def classify(self, error: BaseException, stage: Any = None) -> dict[str, Any]:
        from .atlas_preparation_job_control import _Withdrawn, _reason
        stage, site = stage or self.stage, _raised_at(error)
        if isinstance(error, OSError) and error.errno == errno.ENOENT and isinstance(error.filename, (str, bytes, os.PathLike)):
            row = {"kind": "missing_input", "stage": stage, "code": self.host(error.filename)}
            return {**row, "raised_at": site} if site else row
        row = {"kind": "refusal" if isinstance(error, (ValueError, _Withdrawn)) else "error", "stage": stage,
               "code": _reason(error), "exception": self.host_text(f"{type(error).__name__}: {error}")[:500]}
        return {**row, "raised_at": site} if site else row

    def note(self, record: dict[str, Any], value: Any) -> str | None:
        if isinstance(value, int):
            return None
        text = os.fsdecode(value)
        if not text.startswith(self.watched + os.sep):
            return None
        host = self.host(text)
        known = host in record
        record.setdefault(host, None)
        return None if known else text

    def note_read(self, value: Any) -> None:
        text = self.note(self.reads, value)
        workspace = str(self.workspace) + os.sep
        if text is None or not text.endswith(".json") or not text.startswith(workspace):
            return
        if text[len(workspace):] not in self.retained:
            return
        try:
            with self._open(text, "rb") as stream:
                value = json.loads(stream.read())
        except (OSError, ValueError):
            return
        keys = _PAID_RECORD_KEYS & value.keys() if isinstance(value, dict) else set()
        if keys:
            self.adopted[self.host(text)] = value.get("schema_version") or value.get("status") or sorted(keys)[0]

    def __enter__(self) -> _Harness:
        from . import clean_plate_removal_analysis_gemini as gemini
        from . import meta_sam31
        from . import paid_resource_admission as admission
        from . import paid_resource_allocator as allocator
        from . import website_atlas_pose as pose
        from . import website_atlas_preview as preview
        from . import website_task_context as control
        from .task_evaluation_supervisor import agents_sdk

        stack = self._stack = ExitStack()
        saved = dict(os.environ)
        stack.callback(lambda: (os.environ.clear(), os.environ.update(saved)))
        # Reservations go through the stubbed WebApp only; clients connect where the guard can name the host.
        for name in ("BLUEPRINT_ATLAS_OPERATOR_ADMISSION_FILE", *_PROXY_ENV):
            os.environ.pop(name, None)
        for module, name, value in (
                (control, "website_webapp_request", self.webapp),
                (admission, "require_paid_resource_admission", self.grant),
                (admission, "_monthly_budget_blockers", lambda _resource_class: []),
                (allocator, "_website_atlas_release_blockers", lambda _experimental: (self.commit, [])),
                (pose, "atlas_api_config", lambda: (self.atlas_base, PLACEHOLDER)),
                (pose, "_safe_transport", self.transport),
                (meta_sam31, "meta_api_key", lambda: PLACEHOLDER),
                (gemini, "_api_key", lambda: (PLACEHOLDER, "replay")),
                (agents_sdk, "_file_based_openai_api_key", lambda: PLACEHOLDER),
                (preview, "_archive", lambda _job, _result: self.stop("external_effect", "archive:" + ARTIFACT_KIND,
                                                                      stage="archive")),
                (preview, "_deliver", self.skip("webapp_delivery")),
                (preview, "_clean_provider", self.skip("provider_cleanup")),
                (preview, "_retract_archive", self.skip("archive_retraction"))):
            _swap(stack, module, name, value)
        failed_step, step = preview._failed_step, preview.advance_preparation_step

        def record_failure(job: Any, error: BaseException, current: int) -> Any:
            self.failure, self.failure_stage = error, self.stage
            return failed_step(job, error, current)

        def tracked_step(job: Any, *, checkpoint: Any, **options: Any) -> Any:
            # The controller's own pre-paid-stage checkpoint names each stage as it is reached.
            def tracked(stage: str, **details: Any) -> Any:
                self.reached(stage)
                return checkpoint(stage, **details)
            return step(job, checkpoint=tracked, **options)
        _swap(stack, preview, "_failed_step", record_failure)
        _swap(stack, preview, "advance_preparation_step", tracked_step)
        self._guard(stack)
        return self

    def __exit__(self, *_exc: Any) -> bool:
        self._stack.close()
        return False

    def _guard(self, stack: ExitStack) -> None:
        connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
        popen, exists, is_file, opener = subprocess.Popen.__init__, pathlib.Path.exists, pathlib.Path.is_file, io.open

        def refuse(target: Any) -> None:
            self.network.append({"target": str(target), "call_site": self.origin()["call_site"]})
            self.stop("error", f"unstubbed_network_call:{target}", call_site=self.network[-1]["call_site"])

        def guarded(original: Any) -> Any:
            def call(sock: socket.socket, address: Any) -> Any:
                if sock.family == getattr(socket, "AF_UNIX", None):
                    return original(sock, address)
                refuse(address[0] if isinstance(address, tuple) else address)
            return call

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host in _LOCAL_HOSTS:
                return resolve(host, *args, **kwargs)
            refuse(os.fsdecode(host))

        def spawn(process: Any, args: Any, *rest: Any, **options: Any) -> None:
            argv = (shlex.split(args) if isinstance(args, str)
                    else [os.fsdecode(args)] if isinstance(args, (bytes, os.PathLike)) else [os.fsdecode(a) for a in args])
            tool = os.path.basename(argv[0]) if argv else ""
            if tool in _NETWORK_TOOLS or (tool == "git" and _GIT_NETWORK & set(argv[1:])):
                refuse("subprocess:" + tool)
            if len(rest) < 10:  # env not passed positionally: children get a proxy that answers nothing
                base = options["env"] if options.get("env") is not None else os.environ
                options["env"] = {**base, **{name: _DEAD_PROXY for name in _PROXY_ENV}}
            popen(process, args, *rest, **options)

        def probed(original: Any) -> Any:
            def probe(path: Any, *args: Any, **kwargs: Any) -> bool:
                found = original(path, *args, **kwargs)
                if not found:
                    self.note(self.absent, path)
                return found
            return probe

        def opened(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            stream = opener(file, mode, *args, **kwargs)
            if isinstance(file, (str, bytes, os.PathLike)) and not set(mode) & set("wax+"):
                self.note_read(file)
            return stream

        _set(stack, socket.socket, "connect", guarded(connect))
        _set(stack, socket.socket, "connect_ex", guarded(connect_ex))
        _set(stack, socket, "getaddrinfo", getaddrinfo)
        _set(stack, socket, "create_connection", lambda address, *_a, **_k: refuse(address[0]))
        _set(stack, subprocess.Popen, "__init__", spawn)
        _set(stack, pathlib.Path, "exists", probed(exists))
        _set(stack, pathlib.Path, "is_file", probed(is_file))
        _set(stack, io, "open", opened)
        _set(stack, builtins, "open", opened)


def _snapshots(workspace: Path, host: Any, configuration: str | None,
               task_context: str | None) -> tuple[Any, Any, dict[str, str]]:
    """The allowance and task context WebApp would answer; a missing one is the path to pull."""
    config_path = Path(configuration) if configuration else workspace / "pipeline/atlas/configuration.json"
    config = _read_json(config_path) if config_path.is_file() else host(config_path)
    pipeline_dir = workspace / "pipeline"
    candidates = ([Path(task_context)] if task_context else
                  [pipeline_dir / "website_task_context.json",
                   *sorted((pipeline_dir / "website_task_context_history").glob("*.json"))])
    values = [(path, _read_json(path)) for path in candidates if path.is_file()]
    wanted = config.get("task_context_digest") if isinstance(config, dict) else None
    chosen = next((row for row in values if isinstance(row[1], dict) and row[1].get("context_digest") == wanted),
                  values[0] if values else (candidates[0], host(candidates[0])))
    return config, chosen[1], {"configuration": host(config_path), "task_context": host(chosen[0])}


def run_steps(*, job_path: Path, capture_root: Path, workspace: Path, prefix: str | None = None,
              configuration: str | None = None, task_context: str | None = None, steps: int | None = None,
              commit: str = "") -> dict[str, Any]:
    """Advance the job's copy at ``workspace`` until it stops; ``replay`` makes the copy and checks the evidence."""
    from . import website_atlas_preview as preview
    from .atlas_preparation_job_control import _reason

    workspace, capture_root = Path(workspace), Path(capture_root)
    path = workspace / "pipeline" / "atlas" / JOB_FILE

    def host(value: Any) -> str:
        text = os.fsdecode(value)
        return text[len(prefix):] if prefix and text.startswith(prefix + os.sep) else text
    config, context, sources = _snapshots(workspace, host, configuration, task_context)
    report: dict[str, Any] = {"snapshot_sources": sources, "retried_in_overlay": False, "steps": [],
                              "first_stop": None, "new_paid_calls": 0}
    if not path.is_file():
        return {**report, "first_stop": {"kind": "missing_input", "stage": None, "code": host(path)}}
    # Production creates the lock with the job; it carries no evidence.
    (path.parent / LOCK_FILE).touch()
    first, limit = None, steps or MAXIMUM_STEPS
    harness = _Harness(capture_root=capture_root, workspace=workspace, prefix=prefix, configuration=config,
                       task_context=context, commit=commit)
    with harness:
        while len(report["steps"]) < limit:
            job = read_job(path)
            if job["state"] in TERMINAL and not job.get("retry_at_epoch"):
                if report["steps"] or report["retried_in_overlay"] or job["state"] not in {"failed", "needs_input"}:
                    break
                try:  # what the operator does once the fix is deployed
                    preview.retry_failed_job(path)
                except ValueError as error:
                    first = {"kind": "refusal", "stage": job.get("stage"), "code": _reason(error)}
                    break
                report["retried_in_overlay"] = True
                job = read_job(path)
            current = int(time.time())
            if max(job["next_attempt_at_epoch"], job.get("retry_at_epoch") or 0) > current:
                job = {**job, "next_attempt_at_epoch": min(job["next_attempt_at_epoch"], current),
                       "retry_at_epoch": job.get("retry_at_epoch") and current}
                write_json(path, job)
            harness.failure = harness.failure_stage = None
            try:
                preview.advance_job(path)
            except ReplayStop as stop:
                first = stop.report()
                break
            except Exception as error:  # noqa: BLE001 - the controller itself refused; that is the finding
                first = harness.classify(error)
                break
            after = read_job(path)
            report["steps"].append({key: after.get(key) for key in ("state", "stage", "reason", "transient_failures")})
            harness.reached(after.get("stage"))
            if harness.stops:  # a stop raised where something absorbed it
                first = harness.stops[0].report()
                break
            if harness.failure is not None:
                first = harness.classify(harness.failure, harness.failure_stage or after.get("stage"))
                break
            if after["state"] != "processing":
                if after["state"] != "ready":
                    first = {"kind": "refusal", "stage": after.get("stage"), "code": after.get("reason")}
                break
            if {k: v for k, v in after.items() if k not in _TICK_FIELDS} == {
                    k: v for k, v in job.items() if k not in _TICK_FIELDS}:
                break
    final, unpulled = read_job(path), harness.unpulled()
    if first is not None and unpulled:
        # A retained record that was not pulled reads as absent: a stop right after one may be that.
        first["unpulled_before"] = unpulled[-10:]
    return {**report, "steps_run": len(report["steps"]), "stages_reached": harness.stages, "first_stop": first,
            "would_buy": harness.would_buy, "refused_spend": harness.refused_spend,
            "adopted": [{"path": key, "record": value} for key, value in harness.adopted.items()],
            "absent_paths_probed": unpulled[:200], "skipped_effects": sorted(set(harness.skipped)),
            "network_attempts": harness.network,
            "final_job": {key: final.get(key) for key in ("state", "stage", "reason", "revision")}}


def _namespaces() -> str:
    return " ".join(os.readlink(f"/proc/self/ns/{kind}") for kind in ("mnt", "net"))


def _namespace_prefix(scratch: Path) -> list[str] | None:
    """``unshare`` with a private mount and network namespace and a bind mount in it, if this host permits."""
    if shutil.which("unshare") is None or shutil.which("mount") is None:
        return None
    command = ["unshare", "--mount", "--net", *([] if os.geteuid() == 0 else ["--map-root-user"])]
    probe = subprocess.run([*command, "mount", "--bind", str(scratch), str(scratch)], capture_output=True, check=False)
    return command if probe.returncode == 0 else None


def replay(job: str | Path, *, configuration: str | None = None, task_context: str | None = None,
           steps: int | None = None, replay_root: str | Path | None = None, overlay: str = "auto",
           keep_overlay: bool = False) -> dict[str, Any]:
    """Replay a pulled job's next controller steps on a scratch copy; the pulled evidence is only read."""
    job_path = Path(job).absolute()
    capture_root = job_path.parents[2]
    code_root = Path(__file__).resolve().parents[2]
    commit, clean = _source_state(code_root)
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": SCHEMA, "job_path": str(job_path), "capture_root": str(capture_root),
        "source_commit": commit, "source_tree_clean": clean, "code_root": str(code_root),
        "job_digest": _file_digest(job_path) if job_path.is_file() else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}
    if not job_path.is_file():
        return {**report, "first_stop": {"kind": "missing_input", "stage": None, "code": str(job_path)},
                "new_paid_calls": 0}
    before = _tree_digest(capture_root)
    root = Path(replay_root or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix=f"atlas-replay-{capture_root.name[:48]}-", dir=root))
    namespace = _namespace_prefix(run_root) if overlay == "auto" else None
    options = dict(configuration=configuration, task_context=task_context, steps=steps, commit=commit)
    try:
        if namespace is not None:
            copy, inner = run_root / "overlay", run_root / "inner_report.json"
            shutil.copytree(capture_root, copy, symlinks=True)
            argv = [*namespace, sys.executable, "-m", "blueprint_pipeline.atlas_preview_replay", "--job", str(job_path),
                    "--inner-overlay", str(copy), "--outer-namespace", _namespaces(), "--report", str(inner),
                    "--source-commit", commit,
                    *(["--configuration", str(Path(configuration).absolute())] if configuration else []),
                    *(["--task-context", str(Path(task_context).absolute())] if task_context else []),
                    *(["--steps", str(steps)] if steps else [])]
            completed = subprocess.run(argv, capture_output=True, text=True, check=False,
                                       env={**os.environ, "PYTHONPATH": str(code_root / "src")})
            core = (_read_json(inner) if inner.is_file() else
                    {"first_stop": {"kind": "error", "stage": None, "code": "replay_namespace_failed",
                                    "exception": completed.stderr[-2000:]}, "new_paid_calls": 0})
            report["overlay"] = {"mode": "bind", "copy": str(copy)}
        else:
            mirror = run_root / "mirror"
            workspace = mirror / capture_root.relative_to(capture_root.anchor)
            shutil.copytree(capture_root, workspace, symlinks=True)
            withdrawal = capture_root.parents[1] / "website_withdrawal"
            if withdrawal.is_dir():
                shutil.copytree(withdrawal, workspace.parents[1] / "website_withdrawal", symlinks=True)
            core = run_steps(job_path=job_path, capture_root=capture_root, workspace=workspace, prefix=str(mirror),
                             **options)
            report["overlay"] = {"mode": "relocated", "copy": str(workspace)}
            # Retained records keep absolute host paths the mirror cannot redirect, so a
            # refusal here may be the replay's own containment check, not the job's stop.
            stop = core.get("first_stop") or {}
            if stop.get("kind") in ("refusal", "error"):
                core = {**core, "first_stop": {"kind": "replay_unsupported", "stage": stop.get("stage"),
                        "code": "relocated_replay_cannot_resolve_retained_absolute_paths", "underlying": stop}}
    finally:
        if not keep_overlay:
            for name in ("overlay", "mirror"):
                shutil.rmtree(run_root / name, ignore_errors=True)
    after = _tree_digest(capture_root)
    report.update(core)
    report.update(overlay={**report["overlay"], "kept": keep_overlay},
                  evidence={"digest_before": before, "digest_after": after, "unchanged": before == after},
                  elapsed_seconds=round(time.time() - started, 3),
                  report_path=str(run_root / "atlas_preview_replay_report.v1.json"))
    write_json(Path(report["report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay an Atlas preview job's next controller steps without paying.")
    parser.add_argument("--job", required=True, help="the pulled <capture_root>/pipeline/atlas/preview_job.json")
    parser.add_argument("--configuration", help="allowance snapshot JSON (default: the job's configuration.json)")
    parser.add_argument("--task-context", help="task context JSON (default: the retained one the allowance names)")
    parser.add_argument("--steps", type=int, help=f"controller steps (default: until the job stops, at most {MAXIMUM_STEPS})")
    parser.add_argument("--report", help="also write the report here")
    parser.add_argument("--replay-root", help="scratch root for the overlay copy (default: the temp directory)")
    parser.add_argument("--relocated", action="store_true", help="run at <run>/mirror/<path> without a namespace")
    parser.add_argument("--keep-overlay", action="store_true")
    parser.add_argument("--inner-overlay", help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", default="", help=argparse.SUPPRESS)
    parser.add_argument("--outer-namespace", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.inner_overlay:
        # Inside the private namespace: the scratch copy stands at the capture root's own path.
        if set(_namespaces().split()) & set(args.outer_namespace.split()) or not args.outer_namespace:
            parser.error("--inner-overlay runs only inside the private namespace replay() creates")
        job_path = Path(args.job).absolute()
        subprocess.run(["mount", "--bind", args.inner_overlay, str(job_path.parents[2])], check=True)
        core = run_steps(job_path=job_path, capture_root=job_path.parents[2], workspace=job_path.parents[2],
                         configuration=args.configuration, task_context=args.task_context, steps=args.steps,
                         commit=args.source_commit)
        write_json(Path(args.report), core)
        return 0
    report = replay(args.job, configuration=args.configuration, task_context=args.task_context, steps=args.steps,
                    replay_root=args.replay_root, overlay="relocated" if args.relocated else "auto",
                    keep_overlay=args.keep_overlay)
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return {"refusal": 3, "missing_input": 4, "error": 2, "replay_unsupported": 5}.get(
        (report.get("first_stop") or {}).get("kind"), 0)


__all__ = ["ReplayStop", "SCHEMA", "main", "replay", "run_steps"]

if __name__ == "__main__":
    raise SystemExit(main())
