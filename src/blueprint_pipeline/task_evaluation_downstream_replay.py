"""Replay a website scene job's post-completion chain against this tree, paying for nothing.

Each production fix costs about an hour before the failing step is reached again,
and the chain that runs once a website scene's preparation launch completes had
never run in production. This runs that chain's real code, in order, on scratch
copies of pulled files, until ``--until`` or its first stop:

1. ``completion``: the progression's join (``reconcile_preparation_completion``)
   adopts the completed launch and validates its publication;
2. ``readiness``: the join retains the robot-free readiness receipt, then queues
   the team half;
3. ``team_packet``: ``team_scene_packet_worker`` compiles the team half, writes
   the derived registry row and republishes the job's offer;
4. ``probe_enqueue``: the probe the compile queued; going on, the probe runner's
   next pass is tried and its paid allocator is the stop;
5. ``screening_plan``: ``enqueue_prepared_scenes`` plans within the job's dollar
   ceiling and queues the plan (never the worker's ``main``, which launches).

``first_stop.kind`` is ``completed`` (``--until`` reached), ``would_buy`` (a paid
call, named and never made), ``external_effect`` (a write the replay cannot stand
in for), ``missing_input`` (``code`` is the file to pull, ``pull`` its input, or
the object to add to ``--mirror``), ``refusal`` (the code's own typed blocker,
with the predicates that fired and ``unpulled_before``), or ``error`` (something
the code does not catch). Pulled files are only read: the chain runs on copies,
every root it writes is under ``<replay-root>/<run>/state``, writes anywhere
else stop, and the inputs' digests are compared before and after. Records that
name absolute host paths are read where they point, so pull those to their own
paths. Pull list and command: docs/OPERATOR_DOOR.md.

Boundaries; no flag lifts them. Sockets stop as ``network:<host>``. Only
read-only ``git`` may run as a subprocess; the paid allocator is ``would_buy``.
WebApp ``task-context`` and the non-creating ``atlas-configuration`` read are
answered from ``--webapp-snapshots/<operation>.json``, else the capture's
retained copies (``webapp_answers`` says which); the job offer is recorded,
never sent; any other operation stops. Published objects are answered from
``--mirror/<bucket>/<key>`` or a byte-identical pulled file. The registry copy's
write roots point at scratch, its salt key is a placeholder and it is re-sealed
(``configuration_rewrites``). The probe lane is on, as its unit sets it; its
host-only launch gates are not evaluated and systemd sandboxing is not modelled.
"""

from __future__ import annotations

import argparse
import builtins
import errno
import functools
import io
import json
import os
import pathlib
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit

from .atlas_preview_replay import ReplayStop, _file_digest, _set, _source_state, _swap, _tree_digest
from .decision_evidence_contracts import canonical_digest

SCHEMA = "task_evaluation_downstream_replay_report.v1"
REPORT_FILENAME = "downstream_replay_report.v1.json"
STAGES = ("completion", "readiness", "team_packet", "probe_enqueue", "screening_plan")
STOP_KINDS = ("completed", "would_buy", "external_effect", "missing_input", "refusal", "error")
# Where the scene progression and team units read the operator registry (blueprint-embodiment-preparation.service).
REGISTRY_HOST_PATH = "/etc/blueprint/team-evaluation-registry.json"
INPUT_LABELS = {"launch_run": "launch-run", "intent_dir": "intent-dir", "capture_dir": "capture-dir",
                "registry": "registry"}
PLACEHOLDER_SALT = b"downstream-replay-placeholder-salt-key-never-a-secret\n"
_ALLOCATOR_MODULE = "blueprint_pipeline.paid_resource_allocator"
_GIT_NETWORK = frozenset({"fetch", "pull", "push", "clone", "ls-remote", "remote", "submodule"})
_PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_DEAD_PROXY = "http://127.0.0.1:9"
_LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "::1"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# Frames that only carry a path to the code that asked for it.
_PLUMBING = frozenset({"pathlib", "os", "genericpath", "posixpath", "shutil", "io", "_io", "tempfile",
                       __name__, "blueprint_pipeline.atlas_preview_replay"})
_EXIT_CODES = {"error": 2, "refusal": 3, "missing_input": 4}


def _fd_path(descriptor: int) -> str | None:
    """The directory an open descriptor names (Linux and macOS), else None."""
    try:
        return os.readlink(f"/proc/self/fd/{descriptor}")
    except OSError:
        pass
    try:
        import fcntl

        value = fcntl.fcntl(descriptor, getattr(fcntl, "F_GETPATH"), b"\0" * 1024)
        return os.fsdecode(value.split(b"\0", 1)[0]) or None
    except (AttributeError, OSError, ValueError):
        return None


def _raised_at(error: BaseException) -> str | None:
    """The innermost three package frames that raised, outermost first."""
    sites, trace = [], error.__traceback__
    while trace is not None:
        module = trace.tb_frame.f_globals.get("__name__", "")
        if module.startswith("blueprint_pipeline.") and module not in _PLUMBING:
            sites.append(f"{module.removeprefix('blueprint_pipeline.')}.{trace.tb_frame.f_code.co_name}:"
                         f"{trace.tb_lineno}")
        trace = trace.tb_next
    return " > ".join(sites[-3:]) or None


def _missing_filename(error: BaseException | None) -> str | None:
    """The path of a file-not-found anywhere in an exception's cause chain."""
    seen = 0
    while error is not None and seen < 8:
        if (isinstance(error, OSError) and error.errno == errno.ENOENT
                and isinstance(error.filename, (str, bytes, os.PathLike))):
            return os.fsdecode(error.filename)
        error, seen = error.__cause__ or error.__context__, seen + 1
    return None


def _made(code: str, error: BaseException) -> bool:
    """Whether a typed blocker string was made from this exception (as the chain's own handlers make them)."""
    text = str(error).strip()
    return (code[:200] == text[:200] or code == f"team_scene_packet_failed:{type(error).__name__}"
            or bool(text) and code.startswith(text[:200]))


def _option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def _allocator_call(argv: list[str]) -> dict[str, Any]:
    """The allocator invocation's command, probe kind and ceilings, named from its argv (no secret is in it)."""
    command = argv[0] if argv and not argv[0].startswith("-") else None
    probe = _option(argv, "--probe-kind")
    return {"code": ":".join(part for part in ("paid_resource_allocator", command, probe) if part),
            "command": command, "provider": _option(argv, "--provider"), "probe_kind": probe,
            "max_spend_usd": _option(argv, "--adp-max-spend-usd"),
            "max_hourly_rate_usd": _option(argv, "--adp-max-hourly-rate-usd"),
            "hard_ttl_seconds": _option(argv, "--adp-hard-ttl-seconds"), "argv": list(argv)}


class _Harness:
    """Every boundary of one replay, and what each saw."""

    def __init__(self, *, run: Path, roots: Mapping[str, tuple[Path, Path | None]] | None = None,
                 mirror: Path | None = None, snapshots: Path | None = None, capture: Path | None = None,
                 capture_copy: Path | None = None, until: str = STAGES[-1], commit: str = "") -> None:
        self.run = Path(os.path.realpath(run))
        # label -> (pulled original, replay copy)
        self.roots = {label: (Path(original), Path(copy) if copy else None)
                      for label, (original, copy) in (roots or {}).items()}
        self.mirror, self.snapshots, self.capture, self.capture_copy = mirror, snapshots, capture, capture_copy
        self.until, self.commit = until, commit
        self.code_root = str(Path(__file__).resolve().parents[2])
        self.stage: str | None = None
        self.stops: list[ReplayStop] = []
        self.would_buy: list[dict[str, Any]] = []
        self.recorded_effects: list[dict[str, Any]] = []
        self.webapp_answers: list[dict[str, Any]] = []
        self.object_reads: list[dict[str, Any]] = []
        self.network: list[dict[str, Any]] = []
        self.not_evaluated: list[str] = []
        self.probe_enqueues: list[dict[str, Any]] = []
        self.absent: dict[str, str | None] = {}
        self.ledger: list[tuple[str | None, BaseException]] = []
        self._sizes: dict[int, list[Path]] | None = None
        self._rmtree = 0

    # ---------------------------------------------------------------- paths

    def inside_run(self, text: str) -> bool:
        return text == str(self.run) or text.startswith(str(self.run) + os.sep)

    def original(self, path: Any) -> str:
        """The pulled path a replay copy stands for; any other path unchanged."""
        text = os.path.abspath(os.fsdecode(path))
        for original, copy in self.roots.values():
            if copy is not None and (text == str(copy) or text.startswith(str(copy) + os.sep)):
                return str(original) + text[len(str(copy)):]
        return text

    def pullable(self, path: Any) -> bool:
        """A path the operator can pull: not the replay's own state and not this tree's own files."""
        text = self.original(path)
        return not (self.inside_run(text) or text == self.code_root or text.startswith(self.code_root + os.sep))

    def label(self, path: Any) -> str | None:
        """``<input>:<relative path>`` for a path inside a pulled input or its copy."""
        text = os.path.abspath(os.fsdecode(path))
        for label, (original, copy) in self.roots.items():
            for base in (copy, original):
                if base is not None and (text == str(base) or text.startswith(str(base) + os.sep)):
                    relative = text[len(str(base)) + 1:]
                    return f"{label}:{relative}" if relative else label
        return None

    def call_site(self) -> str | None:
        frame = sys._getframe(1)
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("blueprint_pipeline.") and module not in _PLUMBING:
                return f"{module.removeprefix('blueprint_pipeline.')}.{frame.f_code.co_name}:{frame.f_lineno}"
            frame = frame.f_back
        return None

    def note_absent(self, path: Any) -> None:
        """Remember a path the package looked for and did not find (the latest last)."""
        frame = sys._getframe(2)
        while frame is not None and frame.f_globals.get("__name__") in _PLUMBING:
            frame = frame.f_back
        if frame is None or not str(frame.f_globals.get("__name__", "")).startswith("blueprint_pipeline."):
            return
        text = os.path.abspath(os.fsdecode(path))
        self.absent.pop(text, None)
        self.absent[text] = self.stage

    def unpulled(self, stage: str | None) -> list[str]:
        """Files the stage looked for in a pulled directory that the pull did not include."""
        rows = []
        for text, seen in self.absent.items():
            if seen != stage or os.path.lexists(text):
                continue
            original = self.original(text)
            if os.path.isdir(os.path.dirname(original)) and self.label(original):
                rows.append(self.label(original) or original)
        return rows

    # ----------------------------------------------------------------- stops

    def stop(self, kind: str, code: str, **detail: Any) -> NoReturn:
        stop = ReplayStop(kind, code, **{"stage": self.stage, **detail})
        self.stops.append(stop)
        raise stop

    def missing(self, stage: str | None, path: Any, **detail: Any) -> dict[str, Any]:
        text = self.original(path)
        row: dict[str, Any] = {"kind": "missing_input", "stage": stage, "code": text}
        if self.label(text):
            row["pull"] = self.label(text)
        return {**row, **detail}

    def exception_for(self, stage: str | None, code: str) -> BaseException | None:
        """The recorded exception a stage's typed blocker was made from."""
        return next((error for seen, error in reversed(self.ledger) if seen == stage and _made(code, error)), None)

    def stage_of(self, code: str, default: str) -> str:
        """The stage whose recorded exception a blocker was made from."""
        return next((seen for seen, error in reversed(self.ledger) if seen and _made(code, error)), default)

    def refusal(self, stage: str, code: str, **detail: Any) -> dict[str, Any]:
        """The code's own blocker, unless it was made from a file or host path the pull did not include."""
        from .fail_closed_blocker_explainer import fired_predicates

        error = self.exception_for(stage, code)
        missing = _missing_filename(error)
        if missing is not None and self.pullable(missing):
            return self.missing(stage, missing, underlying_refusal=code)
        hosts = [text for text, seen in self.absent.items() if seen == stage and not self.inside_run(text)
                 and self.pullable(text) and not os.path.lexists(text)]
        if hosts:
            return self.missing(stage, hosts[-1], underlying_refusal=code)
        row: dict[str, Any] = {"kind": "refusal", "stage": stage, "code": code,
                               **{key: value for key, value in detail.items() if value is not None}}
        if error is not None:
            row.update(exception=f"{type(error).__name__}: {error}"[:300], raised_at=_raised_at(error))
            try:
                fired = fired_predicates(error)
            except Exception:  # noqa: BLE001 - explaining must never replace the finding
                fired = []
            if fired:
                row["fired_predicates"] = fired
        unpulled = self.unpulled(stage)
        if unpulled:
            row["unpulled_before"] = unpulled[-10:]
        return row

    def classify(self, error: BaseException) -> dict[str, Any]:
        """A failure the code did not catch: a file it could not find, else an error."""
        missing = _missing_filename(error)
        if missing is not None and self.pullable(missing):
            return self.missing(self.stage, missing, raised_at=_raised_at(error))
        row = {"kind": "error", "stage": self.stage, "code": type(error).__name__,
               "exception": f"{type(error).__name__}: {error}"[:500], "raised_at": _raised_at(error)}
        unpulled = self.unpulled(self.stage)
        return {**row, "unpulled_before": unpulled[-10:]} if unpulled else row

    def run_stage(self, stage: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any] | None]:
        self.stage = stage
        try:
            value = function()
        except ReplayStop as stop:
            return None, stop.report()
        except Exception as error:  # noqa: BLE001 - the code did not catch it; that is the finding
            return None, self.classify(error)
        if self.stops:  # a stop raised where something absorbed it
            return None, self.stops[0].report()
        return value, None

    def recorded(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """``function`` unchanged, except an exception it raises is kept for the stage's explanation."""

        @functools.wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return function(*args, **kwargs)
            except ReplayStop:
                raise
            except BaseException as error:
                self.ledger.append((self.stage, error))
                raise
        return call

    # ------------------------------------------------------------ boundaries

    def _same_size(self, size: Any) -> list[Path]:
        """Pulled files of one size: a published object may be retained byte for byte beside its launch."""
        if self._sizes is None:
            self._sizes = {}
            bases = [self.mirror, *(original for original, _copy in self.roots.values())]
            for base in bases:
                if base is None or not base.is_dir():
                    continue
                for directory, _names, files in os.walk(base):
                    for name in files:
                        path = Path(directory, name)
                        if not path.is_symlink():
                            self._sizes.setdefault(path.stat().st_size, []).append(path)
        return list(self._sizes.get(size, [])) if isinstance(size, int) else []

    def fetch(self, reference: Mapping[str, Any]) -> bytes:
        """A published object from the mirror or a byte-identical pulled file, else the object to pull."""
        from .task_evaluation_configured_controls_deferred_inputs import MAX_DOCUMENT_BYTES

        uri, digest, size = str(reference.get("uri") or ""), reference.get("digest"), reference.get("size_bytes")
        parsed = urlsplit(uri)
        mirror = self.mirror / parsed.netloc / parsed.path.lstrip("/") if self.mirror and parsed.netloc else None
        row: dict[str, Any] = {"uri": uri, "digest": digest, "size_bytes": size, "stage": self.stage,
                               "answered_from": None}
        self.object_reads.append(row)
        for path in [*([mirror] if mirror else []), *self._same_size(size)]:
            if (not os.path.isfile(path) or os.path.islink(path) or os.path.getsize(path) != size
                    or _file_digest(path) != digest):
                continue
            row["answered_from"] = str(path)
            with open(path, "rb") as stream:
                # The object-store reader reads no more than this, so an oversized object fails as it does there.
                return stream.read(min(MAX_DOCUMENT_BYTES, size) + 1)
        self.stop("missing_input", uri, digest=digest, size_bytes=size,
                  **({"mirror_path": str(mirror)} if mirror else {}), call_site=self.call_site())

    def _snapshot(self, operation: str) -> tuple[dict[str, Any], str]:
        retained = {"task-context": "pipeline/website_task_context.json",
                    "atlas-configuration": "pipeline/atlas/configuration.json"}[operation]
        candidates = [*([self.snapshots / f"{operation}.json"] if self.snapshots else []),
                      *([self.capture / retained] if self.capture else [])]
        for path in candidates:
            if os.path.isfile(path):
                value = json.loads(Path(path).read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value, str(path)
        self.stop("missing_input", str(candidates[0]) if candidates else f"webapp-snapshots/{operation}.json",
                  operation=operation, pull="the WebApp's current answer (--webapp-snapshots)",
                  call_site=self.call_site())

    def webapp(self, *, capture_id: str, operation: str, payload: Mapping[str, Any],
               source_kind: str = "website") -> dict[str, Any]:
        if operation == "agent-execution-offer":
            return self.offer(capture_id=capture_id, operation=operation, payload=payload)
        if operation == "task-context" or (operation == "atlas-configuration" and payload.get("create") is False):
            value, source = self._snapshot(operation)
            self.webapp_answers.append({"operation": operation, "capture_id": capture_id, "stage": self.stage,
                                        "answered_from": source, "call_site": self.call_site()})
            # A non-creating read never grants: the WebApp answers it with creation refused.
            return {**value, "grant_creation_allowed": False} if operation == "atlas-configuration" else value
        self.stop("external_effect", "webapp:" + operation, capture_id=capture_id, source_kind=source_kind,
                  call_site=self.call_site())

    def offer(self, *, capture_id: str, operation: str, payload: Mapping[str, Any], **_options: Any) -> dict[str, Any]:
        """Record the job offer the WebApp would receive; it is never sent."""
        offer = dict(payload).get("offer") or {}
        try:
            digest = canonical_digest(dict(payload))
        except (TypeError, ValueError):
            digest = None
        self.recorded_effects.append({
            "boundary": "webapp", "operation": operation, "capture_id": capture_id, "stage": self.stage,
            "sent": False, "payload_digest": digest,
            "team_evaluation_listed": bool(isinstance(offer, Mapping) and offer.get("team_evaluation")),
            "call_site": self.call_site()})
        return {"status": "recorded_not_sent"}

    def bought(self, boundary: str, code: str, **detail: Any) -> NoReturn:
        """A paid call about to happen: recorded, named and stopped."""
        row = {"boundary": boundary, "code": code, **detail, "stage": self.stage, "call_site": self.call_site()}
        self.would_buy.append(row)
        self.stop("would_buy", code, **{key: value for key, value in row.items() if key not in {"code", "stage"}})

    def allocator(self, argv: list[str], **_options: Any) -> NoReturn:
        """The probe runner's canonical allocator: the paid run it would start."""
        self.bought("rigid_robot_free_probe_runner.run_allocator", **_allocator_call([str(value) for value in argv]))

    def admission(self, admission: Mapping[str, Any], *, resource_class: str, **_options: Any) -> NoReturn:
        self.bought("paid_resource_admission", f"paid_resource_admission:{resource_class}",
                    resource_class=resource_class,
                    **{key: admission.get(key) for key in ("provider", "maximum_cost_usd", "request_count")})

    def host_gate(self, name: str) -> Callable[..., list[str]]:
        def not_evaluated(*_args: Any, **_kwargs: Any) -> list[str]:
            if name not in self.not_evaluated:
                self.not_evaluated.append(name)
            return []
        return not_evaluated

    # --------------------------------------------------------------- guards

    def _check_spawn(self, argv: list[str]) -> None:
        tool = os.path.basename(argv[0]) if argv else ""
        index = next((i for i, value in enumerate(argv)
                      if value == _ALLOCATOR_MODULE or value.endswith("paid_resource_allocator.py")), None)
        if index is not None:
            self.bought("subprocess:paid_resource_allocator", **_allocator_call(argv[index + 1:]))
        if tool == "git" and not _GIT_NETWORK & set(argv[1:]):
            return
        target = f"git {' '.join(sorted(_GIT_NETWORK & set(argv[1:])))}" if tool == "git" else tool
        self.network.append({"target": "subprocess:" + target, "stage": self.stage, "call_site": self.call_site()})
        # A child process is outside every stub here: only read-only git may run.
        self.stop("external_effect", "subprocess:" + target, call_site=self.call_site())

    def writable(self, path: Any, *, dir_fd: int | None = None, follow: bool = True) -> None:
        """Admit a write only inside the replay's own run directory."""
        if isinstance(path, int):
            return  # an open descriptor: its file was admitted when it was opened
        text = os.fsdecode(path)
        if dir_fd is not None and not os.path.isabs(text):
            base = _fd_path(dir_fd)
            if base is None:
                if self._rmtree:
                    return  # inside a tree removal already admitted at its root
                self.stop("external_effect", "write_outside_replay_root:<descriptor-relative>:" + text,
                          call_site=self.call_site())
            text = os.path.join(base, text)
        text = os.path.abspath(text)
        real = (os.path.realpath(text) if follow
                else os.path.join(os.path.realpath(os.path.dirname(text)), os.path.basename(text)))
        if real == os.devnull or self.inside_run(real):
            return
        self.stop("external_effect", "write_outside_replay_root:" + text, call_site=self.call_site())

    def _guard(self, stack: ExitStack) -> None:
        connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
        popen = subprocess.Popen.__init__
        probes = {name: getattr(pathlib.Path, name) for name in ("exists", "is_file", "is_dir")}
        opener = io.open
        calls = {name: getattr(os, name) for name in ("open", "mkdir", "replace", "rename", "link", "symlink",
                                                     "unlink", "remove", "rmdir", "chmod", "utime", "truncate")}
        rmtree = shutil.rmtree

        def refuse(target: Any) -> NoReturn:
            site = self.call_site()
            self.network.append({"target": str(target), "stage": self.stage, "call_site": site})
            self.stop("external_effect", f"network:{target}", call_site=site)

        def guarded(original: Any) -> Any:
            def call(sock: socket.socket, address: Any) -> Any:
                if sock.family == getattr(socket, "AF_UNIX", None):
                    return original(sock, address)
                refuse(address[0] if isinstance(address, tuple) else address)
            return call

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host in _LOCAL_HOSTS:
                return resolve(host, *args, **kwargs)
            refuse(os.fsdecode(host) if isinstance(host, (str, bytes)) else host)

        def spawn(process: Any, args: Any, *rest: Any, **options: Any) -> None:
            argv = (shlex.split(args) if isinstance(args, str)
                    else [os.fsdecode(args)] if isinstance(args, (bytes, os.PathLike))
                    else [os.fsdecode(value) if isinstance(value, (str, bytes, os.PathLike)) else str(value)
                          for value in args])
            self._check_spawn(argv)
            if len(rest) < 10:  # env not passed positionally: a child gets a proxy that answers nothing
                base = options["env"] if options.get("env") is not None else os.environ
                options["env"] = {**base, **{name: _DEAD_PROXY for name in _PROXY_ENV}}
            popen(process, args, *rest, **options)

        def probed(original: Any) -> Any:
            def probe(path: Any, *args: Any, **kwargs: Any) -> bool:
                found = original(path, *args, **kwargs)
                if not found and not os.path.lexists(path):  # not a file because it is a directory is not absent
                    self.note_absent(path)
                return found
            return probe

        def opened(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if not isinstance(file, (str, bytes, os.PathLike)):
                return opener(file, mode, *args, **kwargs)
            writing = bool(set(mode) & set("wax+"))
            if writing:
                self.writable(file)
            try:
                return opener(file, mode, *args, **kwargs)
            except FileNotFoundError:
                if not writing:
                    self.note_absent(file)
                raise

        def os_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
            if flags & _WRITE_FLAGS:
                self.writable(path, dir_fd=dir_fd)
            return calls["open"](path, flags, mode, dir_fd=dir_fd)

        def os_mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
            if not (dir_fd is None and os.path.isdir(path)):  # an existing directory only raises FileExistsError
                self.writable(path, dir_fd=dir_fd, follow=False)
            return calls["mkdir"](path, mode, dir_fd=dir_fd)

        def moved(name: str) -> Any:
            def call(src: Any, dst: Any, *args: Any, src_dir_fd: int | None = None,
                     dst_dir_fd: int | None = None, **kwargs: Any) -> Any:
                if name != "link":
                    self.writable(src, dir_fd=src_dir_fd, follow=False)
                self.writable(dst, dir_fd=dst_dir_fd, follow=False)
                return calls[name](src, dst, *args, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, **kwargs)
            return call

        def symlink(src: Any, dst: Any, *args: Any, dir_fd: int | None = None, **kwargs: Any) -> None:
            self.writable(dst, dir_fd=dir_fd, follow=False)
            return calls["symlink"](src, dst, *args, dir_fd=dir_fd, **kwargs)

        def single(name: str) -> Any:
            def call(path: Any, *args: Any, dir_fd: int | None = None, **kwargs: Any) -> Any:
                self.writable(path, dir_fd=dir_fd, follow=name in {"chmod", "utime"})
                return calls[name](path, *args, dir_fd=dir_fd, **kwargs)
            return call

        def truncate(path: Any, length: int) -> None:
            self.writable(path)
            return calls["truncate"](path, length)

        def removed(path: Any, *args: Any, **kwargs: Any) -> None:
            self.writable(path, follow=False)
            self._rmtree += 1
            try:
                return rmtree(path, *args, **kwargs)
            finally:
                self._rmtree -= 1

        _set(stack, socket.socket, "connect", guarded(connect))
        _set(stack, socket.socket, "connect_ex", guarded(connect_ex))
        _set(stack, socket, "getaddrinfo", getaddrinfo)
        _set(stack, socket, "create_connection", lambda address, *_a, **_k: refuse(address[0]))
        _set(stack, subprocess.Popen, "__init__", spawn)
        for name, original in probes.items():
            _set(stack, pathlib.Path, name, probed(original))
        _set(stack, io, "open", opened)
        _set(stack, builtins, "open", opened)
        _set(stack, os, "open", os_open)
        _set(stack, os, "mkdir", os_mkdir)
        for name in ("replace", "rename", "link"):
            _set(stack, os, name, moved(name))
        _set(stack, os, "symlink", symlink)
        for name in ("unlink", "remove", "rmdir", "chmod", "utime"):
            _set(stack, os, name, single(name))
        _set(stack, os, "truncate", truncate)
        _set(stack, shutil, "rmtree", removed)

    def __enter__(self) -> _Harness:
        stack = self._stack = ExitStack()
        saved, bytecode, temporary = dict(os.environ), sys.dont_write_bytecode, tempfile.tempdir
        stack.callback(lambda: (os.environ.clear(), os.environ.update(saved)))
        stack.callback(setattr, sys, "dont_write_bytecode", bytecode)
        stack.callback(setattr, tempfile, "tempdir", temporary)
        scratch = self.run / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        for name in (*_PROXY_ENV, "PIPELINE_SYNC_WEBAPP_URL", "BLUEPRINT_ATLAS_OPERATOR_ADMISSION_FILE"):
            os.environ.pop(name, None)
        os.environ["TMPDIR"] = tempfile.tempdir = str(scratch)
        sys.dont_write_bytecode = True
        try:
            self._install(stack)
        except BaseException:
            stack.close()
            raise
        return self

    def _install(self, stack: ExitStack) -> None:
        from . import beta_policy_screening_capabilities as capabilities
        from . import beta_policy_screening_coordinator as coordinator
        from . import beta_policy_screening_family_sources as family_sources
        from . import beta_policy_screening_job_ceiling as job_ceiling
        from . import beta_policy_screening_worker as screening
        from . import paid_resource_admission as paid_admission
        from . import rigid_robot_free_probe_lane as lane
        from . import rigid_robot_free_probe_runner as runner
        from . import robot_free_readiness_retention as retention
        from . import task_evaluation_configured_controls_deferred_inputs as deferred_inputs
        from . import task_evaluation_configured_controls_progression_worker as progression_worker
        from . import task_evaluation_scene_attempt_binding as attempt_binding
        from . import task_evaluation_scene_intake as intake
        from . import team_candidate_quick10_registry as registry
        from . import team_scene_packet_worker as packets
        from . import website_task_context as control

        retain, enqueue, captures = (retention.retain_robot_free_readiness, runner.enqueue_rigid_robot_free_probe,
                                     family_sources._profile_captures)

        def retained(**kwargs: Any) -> Any:
            if self.until == "completion":
                # The join adopted the launch and validated its publication: completion is reached.
                self.stop("completed", "until_completion", stage="completion")
            self.stage = "readiness"
            try:
                value = retain(**kwargs)
            except ReplayStop:
                raise
            except BaseException as error:
                self.ledger.append(("readiness", error))
                raise
            self.stage = "completion"
            return value

        def probe_enqueued(capture_root: Any, **kwargs: Any) -> Any:
            value = enqueue(capture_root, **kwargs)
            self.probe_enqueues.append({"stage": self.stage, "call_site": self.call_site(),
                                        **{key: value.get(key) for key in ("status", "queued", "reason",
                                                                           "probe_request_digest")}})
            return value

        def this_capture() -> list[str]:
            # Other captures' registered family scenes are outside this job's replay.
            return [capture for capture in captures() if self.capture_copy and capture == str(self.capture_copy)]

        for module, name, value in (
                (control, "website_webapp_request", self.webapp),
                (deferred_inputs, "default_reference_fetcher", self.fetch),
                (paid_admission, "require_paid_resource_admission", self.admission),
                (runner, "run_allocator", self.allocator),
                (lane, "driver_ceiling_blockers", self.host_gate("driver_ceiling_blockers")),
                (lane, "launch_environment_blockers", self.host_gate("launch_environment_blockers")),
                (retention, "retain_robot_free_readiness", retained),
                (runner, "enqueue_rigid_robot_free_probe", probe_enqueued),
                (family_sources, "_profile_captures", this_capture)):
            _swap(stack, module, name, value)
        for module, names in ((progression_worker, ("_validate_source",)),
                              (attempt_binding, ("require_scene_execution_binding",)),
                              (intake, ("_require", "_read")),
                              (packets, ("_compile", "resolve_capture_root", "packet_inputs", "validated_request")),
                              (registry, ("profile_for_capture",)),
                              (runner, ("_process_entry",)),
                              (screening, ("require_current_source",)),
                              (capabilities, ("build_planning_inputs",)),
                              (job_ceiling, ("plan_within_job_ceiling",)),
                              (coordinator, ("enqueue_screening_campaign",))):
            for name in names:
                _swap(stack, module, name, self.recorded(getattr(module, name)))
        self._guard(stack)

    def __exit__(self, *_exc: Any) -> bool:
        self._stack.close()
        return False


# ------------------------------------------------------------------- inputs


def _capture_identity(capture: Path) -> tuple[str, str]:
    """The capture's scene and capture ids: its descriptor, else its place in the partition."""
    try:
        descriptor = json.loads((capture / "capture_descriptor.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        descriptor = {}
    scene = str(descriptor.get("scene_id") or "") if isinstance(descriptor, dict) else ""
    capture_id = str(descriptor.get("capture_id") or "") if isinstance(descriptor, dict) else ""
    return scene or capture.parent.parent.name, capture_id or capture.name


def _rewrite_registry(source: Path, target: Path, *, run: Path, capture: Path | None,
                      capture_copy: Path | None, launch: Path, launch_copy: Path) -> list[dict[str, Any]]:
    """The operator registry with its write roots in scratch and this job's rows on the copies, re-sealed.

    A registry whose seal is already broken keeps it broken, so the replay refuses it as production does.
    """
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    rewrites: list[dict[str, Any]] = []
    if not isinstance(value, dict):
        target.write_bytes(raw)  # unchanged: production's own reader refuses it
        target.chmod(0o644)
        return rewrites
    sealed = value.get("registry_digest") == canonical_digest(value, digest_field="registry_digest")

    def change(row: dict[str, Any], key: str, field: str, new: str) -> None:
        rewrites.append({"file": "registry", "field": field, "host": row[key], "replay": new})
        row[key] = new

    derived = value.get("derived_profiles")
    if isinstance(derived, dict):
        if derived.get("root"):
            change(derived, "root", "derived_profiles.root", str(run / "state" / "team-registry-derived"))
        packet = derived.get("packet_inputs")
        if isinstance(packet, dict) and packet.get("output_root"):
            change(packet, "output_root", "derived_profiles.packet_inputs.output_root",
                   str(run / "state" / "team-scene-halves"))
        if derived.get("scored_matrix_salt_key_path"):
            key = run / "inputs" / "registry" / "placeholder-salt.key"
            key.write_bytes(PLACEHOLDER_SALT)
            key.chmod(0o600)
            change(derived, "scored_matrix_salt_key_path", "derived_profiles.scored_matrix_salt_key_path", str(key))
    identity = _capture_identity(capture) if capture is not None else None
    suffix = os.sep + os.path.join("scenes", identity[0], "captures", identity[1]) if identity else None
    for index, row in enumerate(value.get("profiles") or []):
        if not isinstance(row, dict):
            continue
        root = str(row.get("capture_root") or "")
        # An operator row names the capture by its host path: the same path, or the same scene and capture.
        if capture_copy is not None and suffix and (root == str(capture) or root.endswith(suffix)):
            change(row, "capture_root", f"profiles[{index}].capture_root", str(capture_copy))
        probe = str(row.get("robot_free_probe_root") or "")
        if probe and (probe == str(launch) or Path(probe).name == launch.name):
            change(row, "robot_free_probe_root", f"profiles[{index}].robot_free_probe_root", str(launch_copy))
    if sealed:
        value["registry_digest"] = canonical_digest(value, digest_field="registry_digest")
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target.chmod(0o644)
    return rewrites


def _stage_inputs(originals: Mapping[str, Path | None], run: Path) -> dict[str, Any]:
    """Copy the pulled inputs under ``<run>/inputs``; every root the chain writes is under ``<run>``."""
    inputs, state = run / "inputs", run / "state"
    launch = originals["launch_run"]
    launch_copy = inputs / "launch-runs" / launch.name
    shutil.copytree(launch, launch_copy, symlinks=True)
    intent_copy = inputs / "intents" / originals["intent_dir"].name
    shutil.copytree(originals["intent_dir"], intent_copy, symlinks=True)
    capture, capture_copy, identity = originals.get("capture_dir"), None, None
    if capture is not None:
        identity = _capture_identity(capture)
        capture_copy = inputs / "partition" / "scenes" / identity[0] / "captures" / identity[1]
        shutil.copytree(capture, capture_copy, symlinks=True)
        # A website withdrawal tombstone lives beside the captures.
        withdrawal = capture.parent.parent / "website_withdrawal"
        if withdrawal.is_dir():
            shutil.copytree(withdrawal, capture_copy.parent.parent / "website_withdrawal", symlinks=True)
    registry, registry_copy, rewrites = originals.get("registry"), None, []
    if registry is not None:
        registry_copy = inputs / "registry" / registry.name
        registry_copy.parent.mkdir(parents=True)
        rewrites = _rewrite_registry(registry, registry_copy, run=run, capture=capture,
                                     capture_copy=capture_copy, launch=launch, launch_copy=launch_copy)
    for name in ("team-scene-packets", "rigid-robot-free-probes", "beta-policy-screening"):
        (state / name).mkdir(parents=True, exist_ok=True)
    return {"launch_copy": launch_copy, "intent_copy": intent_copy, "capture_copy": capture_copy,
            "registry_copy": registry_copy, "partition": inputs / "partition" if capture_copy else None,
            "packet_queue": state / "team-scene-packets", "probe_queue": state / "rigid-robot-free-probes",
            "screening_queue": state / "beta-policy-screening", "configuration_rewrites": rewrites}


def _environment(staged: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """What the deployed units set, with every root pointed at the replay; and what is left unset."""
    from .beta_policy_candidate_catalog import CATALOG_ENV
    from .beta_policy_screening_coordinator import QUEUE_ROOT_ENV as SCREENING_ENV
    from .rigid_robot_free_probe_lane import ENABLE_ENV
    from .rigid_robot_free_probe_runner import QUEUE_ENV as PROBE_ENV
    from .team_candidate_registry_reader import REGISTRY_ENV
    from .team_scene_packet_worker import CAPTURE_PARTITION_ENV, QUEUE_ROOT_ENV as PACKET_ENV

    values = {PACKET_ENV: str(staged["packet_queue"]), PROBE_ENV: str(staged["probe_queue"]),
              SCREENING_ENV: str(staged["screening_queue"]), ENABLE_ENV: "1"}
    if staged["partition"] is not None:
        values[CAPTURE_PARTITION_ENV] = str(staged["partition"])
    if staged["registry_copy"] is not None:
        values[REGISTRY_ENV] = str(staged["registry_copy"])
    # The reviewed catalog in this tree, as a deploy of it would install.
    unset = [CATALOG_ENV, *(name for name in (CAPTURE_PARTITION_ENV, REGISTRY_ENV) if name not in values)]
    return values, unset


def _digest(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return _tree_digest(path) if path.is_dir() else _file_digest(path)


# -------------------------------------------------------------------- chain


def _plan_summary(queue: Path, campaign_id: str) -> dict[str, Any]:
    path = queue / "plans" / f"{campaign_id}.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "campaign_id": campaign_id, "plan_digest": plan.get("plan_digest"),
            "budget_usd": plan.get("budget_usd"), "budget_basis": plan.get("budget_basis"),
            "maximum_planned_cost_usd": plan.get("maximum_planned_cost_usd"),
            "candidates": [{key: row.get(key) for key in ("candidate_id", "status", "blockers", "max_cost_usd")}
                           for row in plan.get("candidates") or []]}


def _progression(directory: Path, intent: Mapping[str, Any]) -> dict[str, Any]:
    """Whether production's progression would call the join: an activation, and not already completed."""
    from .task_evaluation_scene_progression_state import load_progression

    if not (directory / "progression-events").is_dir():
        return {"status": "not_pulled"}
    try:
        progress = load_progression(directory, intent)
    except (OSError, ValueError, KeyError) as exc:
        return {"status": "unreadable", "reason": str(exc)[:200]}
    if progress is None:
        return {"status": "none"}
    return {"status": progress.get("status"), "phase": progress.get("phase"),
            "activation_recorded": bool((progress.get("state") or {}).get("activation"))}


def _run_chain(harness: _Harness, staged: Mapping[str, Any], *, until: str,
               capture_given: bool, registry_given: bool) -> dict[str, Any]:
    from . import beta_policy_screening_worker as screening
    from . import rigid_robot_free_probe_runner as runner
    from . import team_scene_packet_worker as packets
    from . import task_evaluation_scene_intake as intake
    from .rigid_robot_free_probe_bundle import current_probe_state
    from .task_evaluation_scene_preparation_completion import reconcile_preparation_completion
    from .team_candidate_quick10_registry import profile_for_capture

    stages: dict[str, dict[str, Any]] = {name: {"stage": name, "status": "not_reached"} for name in STAGES}

    def passed(stage: str, **outcome: Any) -> None:
        stages[stage] = {"stage": stage, "status": "passed", "outcome": outcome}

    def end(stop: dict[str, Any]) -> dict[str, Any]:
        """Every stage before the stop passed; the stop's own stage stopped unless it was ``--until``."""
        stage = stop.get("stage")
        if stage in stages:
            for name in STAGES[:STAGES.index(stage)]:
                if stages[name]["status"] == "not_reached":
                    stages[name]["status"] = "passed"
            if stop["kind"] == "completed":
                stages[stage]["status"] = "passed"
            else:
                stages[stage]["status"] = "stopped"
        return {"stages": list(stages.values()), "first_stop": stop}

    def completed(stage: str, **detail: Any) -> dict[str, Any]:
        return end({"kind": "completed", "stage": stage, "code": f"until_{stage}", **detail})

    # 1-2. The join: completion, then readiness retention inside it.
    intent_path = staged["intent_copy"] / "intent.json"
    harness.stage = "completion"
    if not intent_path.is_file():
        return end(harness.missing("completion", intent_path))
    try:
        intent = intake._read(intent_path, "intent_digest")
    except (OSError, ValueError, KeyError) as exc:
        return end(harness.refusal("completion", str(exc)[:200] or type(exc).__name__))
    config = {"intent_root": str(staged["intent_copy"].parent),
              "launch_execution_root": str(staged["launch_copy"].parent)}
    joined, stop = harness.run_stage("completion", lambda: reconcile_preparation_completion(
        intent=intent, config=config, fetcher=harness.fetch))
    progression = _progression(staged["intent_copy"], intent)
    if stop is not None:
        passed("completion", progression=progression)
        return end(stop)
    if joined is None:
        return end(harness.refusal("completion", "preparation_completion_not_adopted",
                                   progression=progression))
    if joined.get("status") != "completed":
        code = str((joined.get("blockers") or ["preparation_completion_" + str(joined.get("status"))])[0])
        stage = harness.stage_of(code, harness.stage or "completion")
        if stage == "readiness":
            passed("completion", progression=progression)
        return end(harness.refusal(stage, code))
    joined_state = joined["state"]["scene_preparation_completion"]
    request = dict(joined_state.get("team_scene_request") or {})
    passed("completion", join_status=joined["status"], phase=joined.get("phase"), team_scene_request=request,
           progression=progression)
    passed("readiness", **dict(joined_state.get("robot_free_readiness") or {}))
    if until == "readiness":
        return completed("readiness")

    # 3. The team half: the request the join queued, compiled by the packet worker.
    harness.stage = "team_packet"
    if request.get("status") not in {"enqueued", "already_enqueued"}:
        return end(harness.refusal("team_packet", str(request.get("reason") or request.get("status")),
                                   team_scene_request=request))
    processed, stop = harness.run_stage("team_packet", lambda: packets.process_team_scene_queue(
        queue_root=staged["packet_queue"], max_messages=1, fetcher=harness.fetch, offer_transport=harness.offer))
    if stop is not None:
        return end(stop)
    status = next((row for row in processed.get("results") or [] if row.get("request_key") == request["request_key"]),
                  None)
    if status is None:
        return end(harness.refusal("team_packet", "team_scene_request_not_processed", run=processed))
    outcome = {key: status.get(key) for key in ("status", "outcome", "reason", "blockers", "offer", "attempt_root",
                                                 "team_scene_inputs_path", "derived_profile",
                                                 "embodiment_manifests_unreadable") if key in status}
    if status.get("status") == "skipped":
        reason = str(status.get("reason"))
        if reason == "team_candidate_registry_not_configured" and not registry_given:
            return end(harness.missing("team_packet", REGISTRY_HOST_PATH, pull="--registry",
                                       underlying_refusal=reason))
        held, stop = harness.run_stage("team_packet", lambda: profile_for_capture(staged["capture_copy"])
                                       if staged["capture_copy"] else None)
        if stop is not None:
            return end(stop)
        if not held:
            return end(harness.refusal("team_packet", reason, team_packet=outcome))
    elif status.get("status") != "completed":
        code = str((status.get("blockers") or ["team_scene_packet_blocked"])[0])
        if not capture_given and code in {"team_scene_capture_partition_not_configured",
                                          "team_scene_capture_root_missing"}:
            return end(harness.missing("team_packet", f"scenes/*/captures/{status.get('capture_id')}",
                                       pull="--capture-dir", underlying_refusal=code))
        return end(harness.refusal("team_packet", code, error=status.get("error")))
    passed("team_packet", **outcome)
    if until == "team_packet":
        return completed("team_packet")

    # 4. The probe the compile queued, and (going on) the runner's next pass.
    capture = staged["capture_copy"]
    probe, stop = harness.run_stage("probe_enqueue", lambda: current_probe_state(capture))
    if stop is not None:
        return end(stop)
    pending = staged["probe_queue"] / "pending"
    entries = sorted(str(path) for path in pending.glob("*.json")) if pending.is_dir() else []
    probe_outcome = {"enqueue": list(harness.probe_enqueues), "probe_state": probe.get("status"),
                     "receipt_status": probe.get("receipt_status"), "queued_entries": entries}
    if until == "probe_enqueue":
        passed("probe_enqueue", **probe_outcome)
        return completed("probe_enqueue")
    commit = harness.commit or None
    rows, stop = harness.run_stage("probe_enqueue", lambda: runner.run_queue(
        staged["probe_queue"], capture_root=capture,
        hooks={"allocator_runner": harness.allocator,
               **({"source_commit": lambda: commit} if commit else {})}))
    if stop is not None:
        stages["probe_enqueue"]["outcome"] = probe_outcome
        return end(stop)
    probe_outcome["runner"] = rows
    for row in rows:
        state = str(row.get("status"))
        if state in {"decided", "receipt_decided"}:
            continue
        if state == "error":
            error = next((error for seen, error in reversed(harness.ledger) if seen == "probe_enqueue"), None)
            return end(harness.classify(error) if error is not None else {
                "kind": "error", "stage": "probe_enqueue", "code": str(row.get("reason"))})
        reason = (row.get("blockers") or [row.get("reason") or (row.get("detail") or {}).get("reason")])[0]
        return end(harness.refusal("probe_enqueue", str(reason or f"rigid_robot_free_probe_{state}"), runner=row))
    passed("probe_enqueue", **probe_outcome)

    # 5. The screening plan, as the screening timer's source pass builds it (never its main).
    sources, stop = harness.run_stage("screening_plan", lambda: screening.enqueue_prepared_scenes(
        queue_root=staged["screening_queue"], packet_queue_root=staged["packet_queue"]))
    if stop is not None:
        return end(stop)
    source = next((row for row in sources.get("sources") or [] if row.get("request_key") == request["request_key"]),
                  None)
    if source is None:
        return end(harness.refusal("screening_plan", "beta_screening_source_not_adopted"))
    if source.get("status") not in {"enqueued", "already_enqueued"}:
        code = str((source.get("blockers") or ["beta_screening_source_blocked"])[0])
        return end(harness.refusal("screening_plan", code, error=source.get("error"),
                                   scene_packet=source.get("scene_packet")))
    passed("screening_plan", **{key: source.get(key) for key in ("status", "campaign_id", "candidate_order",
                                                               "robot_free_probe_status")})
    return completed("screening_plan", plan=_plan_summary(staged["screening_queue"], source["campaign_id"]))


# ------------------------------------------------------------------- replay


def replay(*, launch_run: str | Path, intent_dir: str | Path, until: str, replay_root: str | Path,
           capture_dir: str | Path | None = None, registry: str | Path | None = None,
           mirror: str | Path | None = None, webapp_snapshots: str | Path | None = None) -> dict[str, Any]:
    """Run the post-completion chain on scratch copies of the pulled files until ``until`` or its first stop."""
    if until not in STAGES:
        raise ValueError(f"until must be one of {', '.join(STAGES)}")
    code_root = Path(__file__).resolve().parents[2]
    commit, clean = _source_state(code_root)
    started = time.time()
    originals: dict[str, Path | None] = {
        "launch_run": Path(launch_run).absolute(), "intent_dir": Path(intent_dir).absolute(),
        "capture_dir": Path(capture_dir).absolute() if capture_dir else None,
        "registry": Path(registry).absolute() if registry else None,
        "mirror": Path(mirror).absolute() if mirror else None,
        "webapp_snapshots": Path(webapp_snapshots).absolute() if webapp_snapshots else None}
    root = Path(replay_root).absolute()
    report: dict[str, Any] = {
        "schema_version": SCHEMA, "until": until, "source_commit": commit, "source_tree_clean": clean,
        "code_root": str(code_root), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "new_paid_calls": 0, "provider_mutation_performed": False}
    for name, path in originals.items():
        if path is not None and root.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"replay_root must not be inside the pulled {name}")
    # The mirror and the snapshots are only looked in: an absent one is empty, and each read it cannot
    # answer names its own path.
    absent = [(name, path) for name, path in originals.items()
              if path is not None and name in INPUT_LABELS and not path.exists()]
    before = {name: _digest(path) for name, path in originals.items()}
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"downstream-replay-{originals['launch_run'].name[:48]}-",
                                dir=root.resolve()))
    report.update(run_root=str(run), report_path=str(run / REPORT_FILENAME))
    if absent:
        name, path = absent[0]
        report.update(stages=[{"stage": stage, "status": "not_reached"} for stage in STAGES],
                      first_stop={"kind": "missing_input", "stage": None, "code": str(path),
                                  "pull": "--" + name.replace("_", "-")})
    else:
        staged = _stage_inputs(originals, run)
        environment, unset = _environment(staged)
        roots = {INPUT_LABELS[name]: (originals[name], staged[copy]) for name, copy in (
            ("launch_run", "launch_copy"), ("intent_dir", "intent_copy"), ("capture_dir", "capture_copy"),
            ("registry", "registry_copy")) if originals[name] is not None}
        harness = _Harness(run=run, roots=roots, mirror=originals["mirror"], snapshots=originals["webapp_snapshots"],
                           capture=originals["capture_dir"], capture_copy=staged["capture_copy"], until=until,
                           commit=commit)
        with harness:
            for name in unset:
                os.environ.pop(name, None)
            os.environ.update(environment)
            core = _run_chain(harness, staged, until=until, capture_given=originals["capture_dir"] is not None,
                              registry_given=originals["registry"] is not None)
        stop = core["first_stop"]
        if stop["kind"] not in {"completed", "missing_input"}:
            unpulled = harness.unpulled(stop.get("stage"))
            if unpulled and "unpulled_before" not in stop:
                stop["unpulled_before"] = unpulled[-10:]
        report.update(core, environment=environment, environment_unset=unset,
                      configuration_rewrites=staged["configuration_rewrites"],
                      would_buy=harness.would_buy, recorded_effects=harness.recorded_effects,
                      webapp_answers=harness.webapp_answers, object_reads=harness.object_reads,
                      network_attempts=harness.network, host_only_gates_not_evaluated=harness.not_evaluated,
                      absent_paths_probed=[harness.label(path) or harness.original(path)
                                           for path in harness.absent if not os.path.lexists(path)][:200])
    after = {name: _digest(path) for name, path in originals.items()}
    report["inputs"] = {name: None if path is None else {
        "path": str(path), "digest_before": before[name], "digest_after": after[name],
        "unchanged": before[name] == after[name]} for name, path in originals.items()}
    report["evidence"] = {"unchanged": all(before[name] == after[name] for name in originals)}
    report["elapsed_seconds"] = round(time.time() - started, 3)
    # Plain JSON throughout, so what is returned is exactly what is written.
    report = json.loads(json.dumps(report, sort_keys=True, default=str))
    Path(report["report_path"]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launch-run", required=True, help="the pulled <launch-runs>/<launch_id> directory")
    parser.add_argument("--intent-dir", required=True, help="the pulled <scene-intake-root>/<intent_id> directory")
    parser.add_argument("--capture-dir", help="the pulled <partition>/scenes/<scene>/captures/<capture> directory")
    parser.add_argument("--registry", help=f"the pulled operator registry (host: {REGISTRY_HOST_PATH})")
    parser.add_argument("--mirror", help="published objects not retained locally, at <mirror>/<bucket>/<key>")
    parser.add_argument("--webapp-snapshots",
                        help="the WebApp's current answers: task-context.json and atlas-configuration.json")
    parser.add_argument("--until", required=True, choices=STAGES)
    parser.add_argument("--replay-root", required=True, help="scratch root; each replay makes its own directory")
    parser.add_argument("--report", help="also write the report here")
    args = parser.parse_args(argv)
    try:
        report = replay(launch_run=args.launch_run, intent_dir=args.intent_dir, capture_dir=args.capture_dir,
                        registry=args.registry, mirror=args.mirror, webapp_snapshots=args.webapp_snapshots,
                        until=args.until, replay_root=args.replay_root)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return _EXIT_CODES.get((report.get("first_stop") or {}).get("kind"), 0)


__all__ = ["REPORT_FILENAME", "ReplayStop", "SCHEMA", "STAGES", "STOP_KINDS", "main", "replay"]

if __name__ == "__main__":
    raise SystemExit(main())
