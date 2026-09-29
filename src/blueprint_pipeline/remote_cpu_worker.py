"""Run one remote CPU stage inside a Cloud Run job execution (plan 14 §4-§6, §9, §10).

``python -m blueprint_pipeline.remote_cpu_worker bootstrap`` is the job's command.  Bootstrap runs
the image's code and trusts nothing it has not checked.  It refuses extra argv (an override would
append them), a task count other than 1, and a transport that is not the pinned generation of the
object the dispatcher named, whose descriptor is not the sealed one named, or any of whose
presigned URLs names another object than the descriptor's or lies outside
``BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX`` (``https://<B2 host>/<bucket>/<key prefix>/``).  Until the
transport is trusted nothing is uploaded; after that every failure is an ``infrastructure_failed``
receipt.  An existing receipt means this execution is a duplicate, and it does nothing.

Bootstrap then checks that every path root takes a new directory (uid 10001 cannot write ``/opt``),
fetches the release source by presigned GET, re-hashes it against the descriptor, checks that it
is the descriptor's commit and holds only recipe v2's paths, extracts it under
``/tmp/blueprint-release/<commit>/`` and starts ``execute`` from that tree with the transport on
its stdin.  No URL is logged, written to disk, or put in argv or an environment variable.

The worker holds no credential beyond the job's own identity, which can only read the transport
object; every other byte moves through the attempt's presigned GETs and PUTs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from . import remote_cpu_job_contract as contract
from .decision_evidence_contracts import canonical_digest
from .remote_cpu_environment import environment_record
from .remote_cpu_job_contract import (
    HEARTBEAT_SCHEMA_VERSION,
    INFRASTRUCTURE_FAILED,
    MAX_RECEIPT_BYTES,
    PHASES,
    PROBE_STAGE,
    RECEIPT_SCHEMA_VERSION,
    STAGES,
    TRANSPORT_SCHEMA_VERSION,
    RemoteCpuContractError,
    record_bytes,
    validate_receipt,
)

MODULE = "blueprint_pipeline.remote_cpu_worker"
# What the dispatcher sets (``cloud_run_jobs_client``) and mints (``remote_cpu_job_allocator``); the
# worker names them itself so that neither module, nor any authority, is in its import closure.
ATTEMPT_VARIABLE = "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID"
DESCRIPTOR_VARIABLE = "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256"
TRANSPORT_OBJECT_VARIABLE = "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT"
TRANSPORT_GENERATION_VARIABLE = "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"
OVERRIDE_VARIABLES = (ATTEMPT_VARIABLE, DESCRIPTOR_VARIABLE, TRANSPORT_OBJECT_VARIABLE, TRANSPORT_GENERATION_VARIABLE)
STAGE_VARIABLE = "BLUEPRINT_REMOTE_CPU_STAGE"
PREFIX_VARIABLE = "BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX"
STAGING_OBJECTS = ("blobs.tar", "index.json", "receipt.json", "heartbeat.json")
# The runtime-source bundle's manifest (``task_evaluation_native_arena_preparation_adapter.MANIFEST_NAME``).
BUNDLE_MANIFEST = "task_evaluation_adapter_bundle_manifest.v1.json"
RELEASE_PARENT = "/tmp/blueprint-release"
PYCACHE_ROOT = "/tmp/blueprint-pycache"
HANDOFF_SCHEMA_VERSION = "remote_cpu_worker_handoff.v1"
STAGE_HANDLERS: Mapping[str, str] = {}
EXIT_REFUSED = 2
TRANSFER_TIMEOUT_SECONDS = 60.0
FINAL_RECEIPT_SECONDS = 60
MAX_TRANSPORT_BYTES = 16 * 1024 * 1024
_CHUNK = 1024 * 1024
_NAME = re.compile(r"[a-z][a-z0-9-]{0,126}[a-z0-9]")
_GENERATION = re.compile(r"[1-9][0-9]{0,19}")
_TRANSPORT_OBJECT = re.compile(r"gs://([a-z0-9][a-z0-9._-]{1,61}[a-z0-9])/(transport/(rcj-[a-z]{2}-[0-9a-f]{24})/"
                               r"(\3-a[1-9][0-9]{0,2}-[0-9a-f]{32})-[0-9a-f]{32}\.json)")
_TRANSPORT_KEYS = frozenset({"schema_version", "descriptor", "inputs", "source_archive", "receipt_url", "outputs",
                             "read_urls_expire_at_epoch", "write_urls_expire_at_epoch"})
_INPUT_KEYS = frozenset({"materialize_at", "digest", "size_bytes", "url"})


class WorkerFailure(RuntimeError):
    """A typed refusal or failure; ``code`` never carries a URL."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


class TransferError(WorkerFailure):
    """A presigned transfer that did not complete; ``code`` is a status or an error type, never the URL."""


def _code(exc: BaseException) -> str:
    return exc.code if isinstance(exc, WorkerFailure) else type(exc).__name__


def _line(mode: str, status: str, code: str | None = None) -> str:
    return json.dumps({"mode": mode, "status": status, **({"code": code} if code else {})})


def _log_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


@dataclass
class WorkerRuntime:
    """What one execution touches: its environment, its presigned transfers, the transport reader and ``/``.

    ``filesystem_root`` stands for ``/``: a descriptor path ``/var/lib/blueprint/x`` lives at
    ``filesystem_root / "var/lib/blueprint/x"``, and the release under ``/tmp`` likewise.
    """

    environ: Mapping[str, str]
    http: Any
    reader: Any = None
    filesystem_root: Path = Path("/")
    clock: Callable[[], float] = time.monotonic
    measure: Callable[[], Mapping[str, Any]] = environment_record
    handlers: Mapping[str, str] = field(default_factory=lambda: dict(STAGE_HANDLERS))
    launch: Callable[[dict[str, Any]], int] | None = None
    run_stage: Callable[..., dict[str, Any]] | None = None
    log: Callable[[str], None] = _log_stderr

    def local(self, path: str) -> Path:
        return Path(self.filesystem_root) / str(path).lstrip("/")


def _refuse(runtime: WorkerRuntime, mode: str, code: str) -> int:
    runtime.log(_line(mode, "refused", code))
    return EXIT_REFUSED


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    digest, size = hashlib.sha256(), 0
    with os.fdopen(os.open(path, flags), "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


class PresignedTransfers:
    """GETs and PUTs through presigned URLs, and only through the audited outbound boundary.

    Every failure is typed by its HTTP status or exception type; no URL reaches a message or a log.
    """

    @staticmethod
    def _open(url: str, *, method: str, data: Any, headers: Mapping[str, str], timeout: float,
              max_bytes: int | None = None) -> bytes | None:
        import urllib.request

        from . import safe_outbound_http as http

        try:
            policy = http.presigned_transfer_policy(url, **({"max_response_bytes": max_bytes} if max_bytes else {}))
            request = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
            return http.open_request(request, policy=policy, timeout_seconds=timeout).body
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code == 404 and method == "GET":
                return None
            raise TransferError(f"http_{exc.code}") from None
        except Exception as exc:  # noqa: BLE001 - the message of a transport error may carry the URL
            raise TransferError(type(exc).__name__) from None

    def read(self, url: str, *, max_bytes: int, timeout: float) -> bytes | None:
        return self._open(url, method="GET", data=None, headers={}, timeout=timeout, max_bytes=max_bytes)

    def download(self, url: str, destination: Path, *, max_bytes: int, timeout: float) -> None:
        from . import safe_outbound_http as http

        try:
            http.download_file_observed(url, output_path=destination, max_bytes=max_bytes, timeout_seconds=timeout,
                                        policy=http.presigned_transfer_policy(url))
        except urllib.error.HTTPError as exc:
            exc.close()
            raise TransferError(f"http_{exc.code}") from None
        except Exception as exc:  # noqa: BLE001 - the message of a transport error may carry the URL
            raise TransferError(type(exc).__name__) from None

    def upload(self, url: str, *, size: int, body: bytes | Callable[[Any], Any], content_type: str,
               timeout: float) -> None:
        """PUT ``body`` (bytes, or a writer streamed through a pipe) with a known ``Content-Length``."""

        headers = {"Content-Type": content_type, "Content-Length": str(size)}
        if isinstance(body, bytes):
            self._open(url, method="PUT", data=body, headers=headers, timeout=timeout)
            return
        read_fd, write_fd = os.pipe()
        failures: list[BaseException] = []

        def produce() -> None:
            try:
                with os.fdopen(write_fd, "wb") as sink:
                    body(sink)
            except BaseException as exc:  # noqa: BLE001 - reported after the request ends
                failures.append(exc)

        producer = threading.Thread(target=produce, name="remote-cpu-upload", daemon=True)
        with os.fdopen(read_fd, "rb") as source:
            producer.start()
            try:
                self._open(url, method="PUT", data=source, headers=headers, timeout=timeout)
            finally:
                source.close()
                producer.join()
        if failures:
            raise TransferError(f"body_{type(failures[0]).__name__}")


class GcsTransportReader:
    """``storage.objects.get`` on the transport bucket, as the job's own identity; the worker holds no key."""

    def __init__(self, bucket: str) -> None:
        self.bucket = bucket

    def get(self, object_name: str, *, generation: int) -> bytes:
        from google.cloud import storage

        blob = storage.Client().bucket(self.bucket).blob(object_name, generation=generation)
        return blob.download_as_bytes(start=0, end=MAX_TRANSPORT_BYTES)


def _sealed_descriptor(value: Any) -> dict[str, Any]:
    """The checks that need no host config: shape, seal, identity, paths and closure (plan 14 §3)."""

    try:
        descriptor = contract._clone(value, "descriptor")
        reasons = contract.forbidden_record_content(descriptor)
        contract._check(descriptor, contract._DESCRIPTOR_SPEC, "", reasons)
        if not reasons and descriptor["descriptor_digest"] != canonical_digest(descriptor,
                                                                                digest_field="descriptor_digest"):
            reasons.append("remote_cpu_descriptor_digest_mismatch")
        if not reasons:
            reasons.extend(contract._identity_reasons(descriptor))
            reasons.extend(contract._closure_reasons(descriptor, contract._descriptor_path_reasons(descriptor, reasons)))
    except RemoteCpuContractError:
        reasons = ["remote_cpu_descriptor_invalid"]
    if reasons:
        raise WorkerFailure("remote_cpu_worker_descriptor_invalid")
    return descriptor


def _object_prefix(value: Any) -> tuple[str, str, str]:
    """``https://<host>/<bucket>/<key prefix>/`` as its scheme, host and path."""

    parts = urlsplit(str(value or ""))
    segments = parts.path.split("/")
    if (parts.scheme != "https" or not parts.hostname or "@" in parts.netloc or parts.query or parts.fragment
            or len(segments) < 3 or segments[-1] != "" or any(part in {"", ".", ".."} for part in segments[1:-1])):
        raise WorkerFailure("remote_cpu_worker_object_prefix_invalid")
    return parts.scheme, parts.netloc, parts.path


def _check_url(url: Any, uri: str, prefix: tuple[str, str, str]) -> None:
    """A presigned URL names exactly ``uri``'s object, on the pinned host, under the pinned key prefix."""

    parts = urlsplit(url) if isinstance(url, str) else None
    path = "/" + uri.removeprefix("s3://")
    if (parts is None or (parts.scheme, parts.netloc) != prefix[:2] or parts.fragment or not path.startswith(prefix[2])
            or unquote(parts.path) != path):
        raise WorkerFailure("remote_cpu_worker_url_outside_prefix")


def validated_transport(transport: Any, environ: Mapping[str, str]) -> dict[str, Any]:
    """The sealed descriptor the dispatcher named, for this job, with one URL per object it names."""

    if (not isinstance(transport, dict) or set(transport) != _TRANSPORT_KEYS
            or transport["schema_version"] != TRANSPORT_SCHEMA_VERSION):
        raise WorkerFailure("remote_cpu_worker_transport_invalid")
    descriptor = _sealed_descriptor(transport["descriptor"])
    if (descriptor["descriptor_digest"], descriptor["attempt_id"]) != (environ.get(DESCRIPTOR_VARIABLE),
                                                                        environ.get(ATTEMPT_VARIABLE)):
        raise WorkerFailure("remote_cpu_worker_descriptor_mismatch")
    stage = str(environ.get(STAGE_VARIABLE) or "").replace("-", "_")
    # A probe runs on the job of the stage it probes (plan 14 §8).
    if (descriptor["execution"]["job"] != environ.get("CLOUD_RUN_JOB") or stage not in STAGES
            or descriptor["stage"] not in {stage, PROBE_STAGE}):
        raise WorkerFailure("remote_cpu_worker_job_mismatch")
    source, inputs, outputs = transport["source_archive"], transport["inputs"], transport["outputs"]
    archive, staging = descriptor["code"]["source_archive"], descriptor["outputs"]["staging_prefix"]
    if (not isinstance(source, dict) or set(source) != {"digest", "size_bytes", "url"}
            or (source["digest"], source["size_bytes"]) != (archive["digest"], archive["size_bytes"])
            or not isinstance(outputs, dict) or set(outputs) != set(STAGING_OBJECTS) or not isinstance(inputs, list)
            or len(inputs) != len(descriptor["inputs"])
            or not all(isinstance(row, dict) and set(row) == _INPUT_KEYS
                       and {name: row[name] for name in ("materialize_at", "digest", "size_bytes")}
                       == {name: item[name] for name in ("materialize_at", "digest", "size_bytes")}
                       for row, item in zip(inputs, descriptor["inputs"]))):
        raise WorkerFailure("remote_cpu_worker_transport_invalid")
    prefix = _object_prefix(environ.get(PREFIX_VARIABLE))
    for url, uri in [(transport["receipt_url"], staging + "receipt.json"),
                     *((outputs[name], staging + name) for name in STAGING_OBJECTS), (source["url"], archive["uri"]),
                     *((row["url"], item["uri"]) for row, item in zip(inputs, descriptor["inputs"]))]:
        _check_url(url, uri, prefix)
    return {**transport, "descriptor": descriptor}


def read_transport(runtime: WorkerRuntime) -> dict[str, Any]:
    """The transport at exactly the generation the dispatcher named: a replaced object is not it (plan 14 §4)."""

    environ = runtime.environ
    match = _TRANSPORT_OBJECT.fullmatch(str(environ.get(TRANSPORT_OBJECT_VARIABLE) or ""))
    generation = str(environ.get(TRANSPORT_GENERATION_VARIABLE) or "")
    if match is None or match.group(4) != environ.get(ATTEMPT_VARIABLE) or _GENERATION.fullmatch(generation) is None:
        raise WorkerFailure("remote_cpu_worker_transport_unpinned")
    try:
        reader = runtime.reader if runtime.reader is not None else GcsTransportReader(match.group(1))
        payload = reader.get(match.group(2), generation=int(generation))
    except Exception:  # noqa: BLE001 - another generation, a denied read: this is not the named transport
        raise WorkerFailure("remote_cpu_worker_transport_unavailable") from None
    try:
        transport = json.loads(payload) if len(payload) <= MAX_TRANSPORT_BYTES else None
    except (TypeError, ValueError):
        transport = None
    return validated_transport(transport, environ)


class _Attempt:
    """One attempt's presigned writes, heartbeats and then the receipt last, and the counts they report."""

    def __init__(self, runtime: WorkerRuntime, transport: Mapping[str, Any], execution_name: str, *,
                 started: float, sequence: int = 0, fetched: int = 0, uploaded: int = 0,
                 phases: Mapping[str, Any] | None = None) -> None:
        self.runtime, self.transport, self.execution_name = runtime, transport, execution_name
        self.descriptor = transport["descriptor"]
        self.started, self.sequence, self.fetched, self.uploaded = started, sequence, fetched, uploaded
        self.phases: dict[str, float | None] = {**dict.fromkeys(PHASES), **dict(phases or {})}
        self.phase, self.environment = "bootstrap", None
        self._lock = threading.Lock()

    def elapsed(self) -> float:
        return round(max(0.0, self.runtime.clock() - self.started), 3)

    def receipt_exists(self) -> bool:
        try:
            return self.runtime.http.read(self.transport["receipt_url"], max_bytes=MAX_RECEIPT_BYTES,
                                          timeout=TRANSFER_TIMEOUT_SECONDS) is not None
        except Exception:  # noqa: BLE001 - whether a receipt exists is unknown: never risk overwriting one
            raise WorkerFailure("remote_cpu_worker_receipt_unreadable") from None

    def count_fetched(self, size: int) -> None:
        with self._lock:
            self.fetched += size

    def put(self, name: str, body: bytes | Callable[[Any], Any], *, size: int | None = None,
            content_type: str = "application/json", timeout: float = TRANSFER_TIMEOUT_SECONDS) -> None:
        length = len(body) if isinstance(body, bytes) else int(size or 0)
        try:
            self.runtime.http.upload(self.transport["outputs"][name], size=length, body=body,
                                     content_type=content_type, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - typed by status or type, never by message
            raise WorkerFailure(f"remote_cpu_worker_upload_failed:{name}:{_code(exc)}") from None
        with self._lock:
            self.uploaded += length

    def beat(self) -> bool:
        """PUT the next heartbeat; a lost one is tolerated, and the host's lease goes stale if they stop."""

        with self._lock:
            self.sequence += 1
            heartbeat = {"schema_version": HEARTBEAT_SCHEMA_VERSION, "attempt_id": self.descriptor["attempt_id"],
                         "execution_name": self.execution_name, "sequence": self.sequence, "phase": self.phase,
                         "elapsed_seconds": self.elapsed(), "bytes_fetched": self.fetched,
                         "bytes_uploaded": self.uploaded}
        try:
            self.put("heartbeat.json", record_bytes(heartbeat))
        except WorkerFailure as exc:
            self.runtime.log(_line(self.phase, "heartbeat_lost", exc.code))
            return False
        return True

    def _environment(self) -> dict[str, Any]:
        if self.environment is None:
            try:
                record = self.runtime.measure()
                self.environment = {"environment_digest": record["environment_digest"], "cpu_class": record["cpu_class"]}
            except Exception as exc:  # noqa: BLE001 - an unmeasured environment never matches the host's
                self.environment = {"environment_digest": canonical_digest({"unmeasured": type(exc).__name__}),
                                    "cpu_class": None}
        return self.environment

    def _receipt(self, status: str, result: Any, output: Any, failures: Sequence[str],
                 misses: Sequence[str]) -> dict[str, Any]:
        environment = self._environment()
        with self._lock:
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                **{name: self.descriptor[name] for name in ("job_id", "attempt", "attempt_id", "stage",
                                                            "descriptor_digest")},
                "execution_name": self.execution_name, "status": status, "result": result, "output": output,
                "infrastructure_failures": sorted(set(failures)), "release_path_misses": sorted(set(misses)),
                "environment": environment, "phases": dict(self.phases), "bytes_fetched": self.fetched,
                "bytes_uploaded": self.uploaded, "private_url_recorded": False, "receipt_digest": "",
            }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        return receipt

    def commit(self, status: str, *, result: Any = None, output: Any = None, failures: Sequence[str] = (),
               misses: Sequence[str] = ()) -> dict[str, Any]:
        """Upload the receipt, the commit marker, last; one the host would refuse becomes an infrastructure failure."""

        receipt = self._receipt(status, result, output, failures, misses)
        try:
            validate_receipt(receipt, descriptor=self.descriptor, execution_name=self.execution_name)
        except RemoteCpuContractError as exc:
            reason = exc.reasons[0] if exc.reasons else "unknown"
            receipt = self._receipt("infrastructure_failed", None, None,
                                    [f"{INFRASTRUCTURE_FAILED}remote_cpu_worker_receipt_invalid:{reason}"], ())
        self.put("receipt.json", record_bytes(receipt))
        return receipt

    def fail(self, code: str, *, mode: str) -> int:
        """Commit an infrastructure-failure receipt; exit 0 once it is up (plan 14 §6)."""

        try:
            self.commit("infrastructure_failed", failures=[INFRASTRUCTURE_FAILED + code])
        except WorkerFailure as exc:
            return _refuse(self.runtime, mode, exc.code)
        self.runtime.log(_line(mode, "infrastructure_failed", code))
        return 0


def _guards(argv: Sequence[str], environ: Mapping[str, str]) -> str:
    if list(argv) != ["bootstrap"]:
        raise WorkerFailure("remote_cpu_worker_argv_invalid")
    if environ.get("CLOUD_RUN_TASK_COUNT") != "1":
        raise WorkerFailure("remote_cpu_worker_task_count_invalid")
    name = str(environ.get("CLOUD_RUN_EXECUTION") or "")
    if _NAME.fullmatch(name) is None:
        raise WorkerFailure("remote_cpu_worker_execution_unnamed")
    return name


def _require_writable_roots(descriptor: Mapping[str, Any], runtime: WorkerRuntime) -> None:
    """Every path root must take a new directory: ``/var/lib/blueprint`` is an in-memory mount (plan 14 §6)."""

    for root in descriptor["limits"]["allowed_path_roots"]:
        local = runtime.local(root)
        probe = local / f".remote-cpu-worker-probe-{secrets.token_hex(8)}"
        try:
            local.mkdir(parents=True, exist_ok=True)
            probe.mkdir()
            probe.rmdir()
        except OSError:
            raise WorkerFailure("remote_cpu_worker_path_root_unwritable") from None


def _download(runtime: WorkerRuntime, attempt: _Attempt, url: str, destination: Path, *, digest: str, size: int,
              mismatch: str, timeout: float) -> None:
    """Fetch by presigned GET into ``destination``, then re-hash it: only the declared bytes survive."""

    try:
        runtime.http.download(url, destination, max_bytes=max(size, 1), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - typed by status or type, never by message
        destination.unlink(missing_ok=True)
        raise WorkerFailure(f"remote_cpu_worker_fetch_failed:{_code(exc)}") from None
    if _hash_file(destination) != (digest, size):
        destination.unlink(missing_ok=True)
        raise WorkerFailure(mismatch)
    attempt.count_fetched(size)


def _recipe_member(member: tarfile.TarInfo) -> bool:
    """A regular file or directory under ``src``, ``docs/schemas`` or ``pyproject.toml`` (recipe v2)."""

    parts = member.name.split("/")
    return ((member.isfile() or member.isdir()) and "\x00" not in member.name
            and all(part not in {"", ".", ".."} for part in parts)
            and (member.name in {"src", "docs", "pyproject.toml"} or parts[0] == "src" or parts[:2] == ["docs", "schemas"]))


def _extract_release(archive: Path, staging: Path, commit: str) -> None:
    try:
        with tarfile.open(archive, "r:") as tar:
            members = tar.getmembers()
            if tar.pax_headers.get("comment") != commit:
                raise WorkerFailure("remote_cpu_worker_source_commit_mismatch")
            if not all(map(_recipe_member, members)):
                raise WorkerFailure("remote_cpu_worker_source_member_invalid")
            staging.mkdir(mode=0o700)
            tar.extractall(staging, members=members, filter="data")
    except (tarfile.TarError, ValueError):
        raise WorkerFailure("remote_cpu_worker_source_invalid") from None


def _fetch_release(transport: Mapping[str, Any], runtime: WorkerRuntime, attempt: _Attempt) -> Path:
    """Fetch recipe v2, verify it, and extract it to ``/tmp/blueprint-release/<commit>/`` (``/opt`` is not writable)."""

    commit, source = transport["descriptor"]["code"]["source_commit"], transport["source_archive"]
    parent = runtime.local(RELEASE_PARENT)
    release, nonce = parent / commit, secrets.token_hex(8)
    archive, staging = parent / f".{commit}.{nonce}.tar", parent / f".{commit}.{nonce}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(release):
            raise WorkerFailure("remote_cpu_worker_release_root_exists")
        _download(runtime, attempt, source["url"], archive, digest=source["digest"], size=source["size_bytes"],
                  mismatch="remote_cpu_worker_source_digest_mismatch", timeout=TRANSFER_TIMEOUT_SECONDS)
        _extract_release(archive, staging, commit)
        os.rename(staging, release)
    except OSError as exc:
        raise WorkerFailure(f"remote_cpu_worker_release_unwritable:{_code(exc)}") from None
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
    return release


def _release_environment(runtime: WorkerRuntime, descriptor: Mapping[str, Any], release: Path) -> dict[str, str]:
    """Execute's environment: the release's code first, the descriptor's variables, and no transport name."""

    environ = {name: value for name, value in runtime.environ.items()
               if name not in {TRANSPORT_OBJECT_VARIABLE, TRANSPORT_GENERATION_VARIABLE}}
    environ.update({name: str(runtime.local(value)) for name, value in descriptor["environment"].items()})
    environ.update(PYTHONPATH=str(release / "src"), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPYCACHEPREFIX=str(runtime.local(PYCACHE_ROOT)))
    return environ


def _wait(process: subprocess.Popen, seconds: float, timeout_code: str) -> int:
    """Wait for a child started in its own session; past ``seconds`` its whole group is killed."""

    try:
        return process.wait(timeout=max(1.0, seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise WorkerFailure(timeout_code) from None


def _launch_execute(runtime: WorkerRuntime, handoff: Mapping[str, Any]) -> int:
    """Start ``execute`` from the extracted release, with the handoff (and its URLs) on its stdin only."""

    release, descriptor = Path(handoff["release_root"]), handoff["transport"]["descriptor"]
    process = subprocess.Popen([sys.executable, "-P", "-m", MODULE, "execute"], cwd=release, stdin=subprocess.PIPE,
                               env=_release_environment(runtime, descriptor, release), start_new_session=True)
    try:
        with process.stdin as stream:
            stream.write(json.dumps(handoff).encode("utf-8"))
    except OSError:
        pass  # execute has already exited; its status says why
    budget = descriptor["limits"]["task_timeout_seconds"] - handoff["elapsed_seconds"] - FINAL_RECEIPT_SECONDS
    return _wait(process, budget, "remote_cpu_worker_execute_timeout")


def bootstrap(argv: Sequence[str], runtime: WorkerRuntime) -> int:
    """The job's command: guard, read the pinned transport, fetch and extract the release, start ``execute``."""

    started = runtime.clock()
    try:
        execution_name = _guards(argv, runtime.environ)
        transport = read_transport(runtime)
    except WorkerFailure as exc:
        return _refuse(runtime, "bootstrap", exc.code)
    os.umask(0o077)
    attempt = _Attempt(runtime, transport, execution_name, started=started)
    try:
        if attempt.receipt_exists():
            runtime.log(_line("bootstrap", "duplicate_execution"))
            return 0
    except WorkerFailure as exc:
        return _refuse(runtime, "bootstrap", exc.code)
    attempt.beat()
    try:
        _require_writable_roots(transport["descriptor"], runtime)
        release = _fetch_release(transport, runtime, attempt)
    except WorkerFailure as exc:
        attempt.phases["bootstrap"] = attempt.elapsed()
        return attempt.fail(exc.code, mode="bootstrap")
    attempt.phases["bootstrap"] = attempt.elapsed()
    handoff = {"schema_version": HANDOFF_SCHEMA_VERSION, "transport": transport, "execution_name": execution_name,
               "filesystem_root": str(runtime.filesystem_root), "release_root": str(release),
               "elapsed_seconds": attempt.elapsed(), "phases": dict(attempt.phases),
               "heartbeat_sequence": attempt.sequence, "bytes_fetched": attempt.fetched,
               "bytes_uploaded": attempt.uploaded}
    try:
        code = runtime.launch(handoff) if runtime.launch is not None else _launch_execute(runtime, handoff)
    except WorkerFailure as exc:
        return attempt.fail(exc.code, mode="bootstrap")
    return 0 if code == 0 else attempt.fail(f"remote_cpu_worker_execute_failed:exit_{code}", mode="bootstrap")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["bootstrap"]:
        return bootstrap(arguments, WorkerRuntime(environ=os.environ, http=PresignedTransfers()))
    return _refuse(WorkerRuntime(environ=os.environ, http=None), "worker", "remote_cpu_worker_argv_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
