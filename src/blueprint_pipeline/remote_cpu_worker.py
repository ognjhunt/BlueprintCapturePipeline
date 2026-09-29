"""Run one remote CPU stage inside a Cloud Run job execution (plan 14 §4-§6, §9, §10).

``bootstrap`` is the job's command and runs the image's code.  It refuses extra argv, a task count
other than 1, and a transport that is not the pinned generation of the named object, whose
descriptor is not the sealed one named, or whose presigned URLs name other objects or leave
``BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX`` (``https://<B2 host>/<bucket>/<key prefix>/``).  Nothing is
uploaded until the transport is trusted; after that a failure is an ``infrastructure_failed``
receipt, and an existing receipt makes the execution a no-op.  It refuses before fetching when a
path root cannot take a new directory, verifies the release archive (digest, commit, recipe v2
paths), extracts it under ``/tmp/blueprint-release/<commit>/`` and starts ``execute`` from it with
the transport on stdin: no URL is ever logged, written to disk, or put in argv or the environment.

``execute`` heartbeats throughout, materializes each input at its declared path (re-hashed, given
its mode, linked into place) and spawns the registered stage from the release, never forking its
threaded self.  The stage child audits reads under the release root: a release path the archive
lacks is an infrastructure failure, never a blocked result.  Past a phase deadline the child's
session is killed and a timeout receipt still uploads.  A changed input, or a write under a path
root outside the output root and declared scratch, is an infrastructure failure.  The output is
sealed with host-known bytes indexed by origin and only new bytes archived; ``receipt.json`` goes
last.  The worker holds no credential beyond the job's own identity, which can only read the
transport object; every other byte moves through the attempt's presigned GETs and PUTs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from . import remote_cpu_job_contract as contract
from .decision_evidence_contracts import canonical_digest
from .remote_cpu_environment import environment_record
from .remote_cpu_job_contract import (
    HEARTBEAT_SCHEMA_VERSION, INFRASTRUCTURE_FAILED, MAX_RECEIPT_BYTES, OUTPUT_FORMAT, PHASES, PROBE_STAGE,
    RECEIPT_SCHEMA_VERSION, RELEASE_PATH_MISSING, STAGES, TRANSPORT_SCHEMA_VERSION, RemoteCpuContractError,
    record_bytes, safe_label, validate_receipt,
)
from .remote_cpu_output_archive import RemoteCpuArchiveError, blobs_tar_size, index_bytes, index_tree, write_blobs_tar
from .remote_cpu_worker_stage import StageRoots as StageRoots  # a handler's roots; re-exported for handlers
from .remote_cpu_worker_stage import from_release, stage_main
from .remote_cpu_worker_stage import install_release_audit as install_release_audit

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
STAGE_REQUEST_SCHEMA_VERSION = "remote_cpu_worker_stage_request.v1"
# ``stage -> "module:function"``; a handler takes the sealed descriptor and ``StageRoots`` and returns the
# stage's sealed result.  An unregistered stage is an infrastructure failure.
STAGE_HANDLERS: Mapping[str, str] = {PROBE_STAGE: "blueprint_pipeline.remote_cpu_worker_stage:run_environment_probe"}
EXIT_REFUSED = 2
TRANSFER_TIMEOUT_SECONDS = 60.0
FINAL_RECEIPT_SECONDS = 60
MAX_TRANSPORT_BYTES = 16 * 1024 * 1024
MAX_HANDOFF_BYTES = 32 * 1024 * 1024
MAX_NAMED_PATHS = 16
_CHUNK = 1024 * 1024
_NAME = re.compile(r"[a-z][a-z0-9-]{0,126}[a-z0-9]")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
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


class _Committed(WorkerFailure):
    """This attempt's receipt is already up; it is never overwritten (plan 14 §6)."""


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
    def _typed(action: Callable[[], Any], *, absent_ok: bool = False) -> Any:
        try:
            return action()
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code == 404 and absent_ok:
                return None
            raise TransferError(f"http_{exc.code}") from None
        except Exception as exc:  # noqa: BLE001 - the message of a transport error may carry the URL
            raise TransferError(type(exc).__name__) from None

    @classmethod
    def _open(cls, url: str, *, method: str, data: Any, headers: Mapping[str, str], timeout: float,
              max_bytes: int | None = None) -> bytes | None:
        import urllib.request

        from . import safe_outbound_http as http

        def send() -> bytes:
            policy = http.presigned_transfer_policy(url, **({"max_response_bytes": max_bytes} if max_bytes else {}))
            request = urllib.request.Request(url, data=data, method=method, headers=dict(headers))
            return http.open_request(request, policy=policy, timeout_seconds=timeout).body

        return cls._typed(send, absent_ok=method == "GET")

    def read(self, url: str, *, max_bytes: int, timeout: float) -> bytes | None:
        return self._open(url, method="GET", data=None, headers={}, timeout=timeout, max_bytes=max_bytes)

    def download(self, url: str, destination: Path, *, max_bytes: int, timeout: float) -> None:
        from . import safe_outbound_http as http

        self._typed(lambda: http.download_file_observed(url, output_path=destination, max_bytes=max_bytes,
                                                        timeout_seconds=timeout,
                                                        policy=http.presigned_transfer_policy(url)))

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
        """Whether a receipt is up; the receipt GET lives as long as the attempt's writes (plan 14 §4)."""
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

    def environment_summary(self) -> dict[str, Any]:
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
        environment = self.environment_summary()
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

    def _refusal(self, receipt: Mapping[str, Any]) -> str | None:
        try:
            validate_receipt(receipt, descriptor=self.descriptor, execution_name=self.execution_name)
        except RemoteCpuContractError as exc:
            return f"{INFRASTRUCTURE_FAILED}remote_cpu_worker_receipt_invalid:{(exc.reasons or ('unknown',))[0]}"
        return None

    def commit(self, status: str, *, result: Any = None, output: Any = None, failures: Sequence[str] = (),
               misses: Sequence[str] = ()) -> dict[str, Any]:
        """Upload the receipt, the commit marker, last, and only while none is up: a committed receipt is never
        overwritten, and when that cannot be read nothing is written.  One the host would refuse keeps its
        failures but not its result, and failing that says only why it was refused."""
        receipt = self._receipt(status, result, output, failures, misses)
        refusal = self._refusal(receipt)
        if refusal is not None:
            receipt = self._receipt("infrastructure_failed", None, None, [*failures, refusal], misses)
            if self._refusal(receipt) is not None:
                receipt = self._receipt("infrastructure_failed", None, None, [refusal], ())
        if self.receipt_exists():
            raise _Committed("receipt_already_committed")
        self.put("receipt.json", record_bytes(receipt))
        return receipt

    def finish(self, status: str, *, mode: str, result: Any = None, output: Any = None, codes: Sequence[str] = (),
               misses: Sequence[str] = ()) -> int:
        """Commit the receipt; exit 0 once it is up (plan 14 §6), or refuse when it could not be."""
        try:
            self.commit(status, result=result, output=output, failures=[INFRASTRUCTURE_FAILED + code for code in codes],
                        misses=misses)
        except _Committed:
            self.runtime.log(_line(mode, "receipt_already_committed"))
            return 0
        except WorkerFailure as exc:
            return _refuse(self.runtime, mode, exc.code)
        self.runtime.log(_line(mode, status, ",".join(codes) or None))
        return 0

    def fail(self, code: str, *, mode: str) -> int:
        return self.finish("infrastructure_failed", mode=mode, codes=[code])


def _guards(argv: Sequence[str], environ: Mapping[str, str]) -> str:
    if list(argv) != ["bootstrap"]:
        raise WorkerFailure("remote_cpu_worker_argv_invalid")
    if environ.get("CLOUD_RUN_TASK_COUNT") != "1":
        raise WorkerFailure("remote_cpu_worker_task_count_invalid")
    name = str(environ.get("CLOUD_RUN_EXECUTION") or "")
    if _NAME.fullmatch(name) is None:
        raise WorkerFailure("remote_cpu_worker_execution_unnamed")
    return name


def _environment_gate(attempt: _Attempt) -> None:
    """Inline NuRec float math needs a qualified CPU class, and any stage but the probe must run in the environment
    it was dispatched for; an unmeasured class is never qualified (plan 14 §5)."""
    environment, descriptor = attempt.environment_summary(), attempt.descriptor
    if (descriptor["closure"]["class"] == "absent_inline_only"
            and environment["cpu_class"] not in descriptor["limits"]["allowed_cpu_classes"]):
        raise WorkerFailure("cpu_class_unqualified")
    if descriptor["stage"] != PROBE_STAGE and environment["environment_digest"] != descriptor["code"]["environment_digest"]:
        raise WorkerFailure("environment_mismatch")


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


class _Deadline:
    def __init__(self, clock: Callable[[], float], seconds: float, phase: str) -> None:
        self.clock, self.seconds, self.phase, self.end = clock, seconds, phase, clock() + seconds

    def check(self) -> None:
        if self.clock() >= self.end:
            raise WorkerFailure(f"phase_deadline:{self.phase}")

    def timeout(self) -> float:
        """A socket timeout that never outlives the phase."""
        self.check()
        return max(1.0, min(TRANSFER_TIMEOUT_SECONDS, self.end - self.clock()))


@contextmanager
def _phase(attempt: _Attempt, name: str) -> Iterator[_Deadline]:
    started, attempt.phase = attempt.runtime.clock(), name
    try:
        yield _Deadline(attempt.runtime.clock, attempt.descriptor["limits"]["phase_seconds"][name], name)
    finally:
        attempt.phases[name] = round(max(0.0, attempt.runtime.clock() - started), 3)


@contextmanager
def _heartbeats(attempt: _Attempt) -> Iterator[None]:
    """Beat at once, then every interval from a thread, and stop before the receipt goes (plan 14 §6)."""
    stopped = threading.Event()

    def run() -> None:
        try:
            while not stopped.wait(attempt.descriptor["limits"]["heartbeat_interval_seconds"]):
                attempt.beat()
        except Exception as exc:  # noqa: BLE001 - a lost heartbeat thread lets the host's lease go stale
            attempt.runtime.log(_line(attempt.phase, "heartbeats_stopped", type(exc).__name__))

    attempt.beat()
    thread = threading.Thread(target=run, name="remote-cpu-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _unchanged(path: Path, item: Mapping[str, Any]) -> bool:
    """A regular file with the input's declared mode, size and digest."""
    try:
        info = os.lstat(path)
        return (stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == int(item["mode"], 8)
                and _hash_file(path) == (item["digest"], item["size_bytes"]))
    except OSError:
        return False


def _materialize_inputs(attempt: _Attempt, runtime: WorkerRuntime, deadline: _Deadline) -> None:
    """Each input by presigned GET into a partial file, re-hashed, given its mode and linked onto ``materialize_at``."""
    for row, item in zip(attempt.transport["inputs"], attempt.descriptor["inputs"]):
        deadline.check()
        target, label = runtime.local(item["materialize_at"]), safe_label(item["materialize_at"])
        if os.path.lexists(target):
            # Only a same-machine run finds an input in place: an identical file is kept, any other refused.
            if not _unchanged(target, item):
                raise WorkerFailure(f"input_conflict:{label}")
            continue
        partial = target.with_name(f".{target.name}.{secrets.token_hex(8)}.partial")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _download(runtime, attempt, row["url"], partial, digest=item["digest"], size=item["size_bytes"],
                      mismatch=f"input_digest_mismatch:{label}", timeout=deadline.timeout())
            os.chmod(partial, int(item["mode"], 8))
            os.link(partial, target)
        except OSError as exc:
            raise WorkerFailure(f"input_unwritable:{label}:{_code(exc)}") from None
        finally:
            partial.unlink(missing_ok=True)
    deadline.check()


def _run_stage_child(*, runtime: WorkerRuntime, descriptor: Mapping[str, Any], release: Path, handler: str,
                     seconds: float) -> dict[str, Any]:
    """Spawn the stage in a fresh interpreter from the release: this parent has threads, so it never forks."""
    runtime.local("/tmp").mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="remote-cpu-stage-", dir=runtime.local("/tmp")))
    try:
        request = {"schema_version": STAGE_REQUEST_SCHEMA_VERSION, "handler": handler, "descriptor": descriptor,
                   "filesystem_root": str(runtime.filesystem_root), "release_root": str(release),
                   "report": str(scratch / "report.json")}
        process = subprocess.Popen([sys.executable, "-P", "-m", MODULE, "stage"], cwd=release, stdin=subprocess.PIPE,
                                   env=_release_environment(runtime, descriptor, release), start_new_session=True)
        try:
            with process.stdin as stream:
                stream.write(json.dumps(request).encode("utf-8"))
        except OSError:
            pass  # the child has already exited; its status says why
        code = _wait(process, seconds, "phase_deadline:stage")
        try:
            with open(scratch / "report.json", "rb") as stream:
                report = json.loads(stream.read(MAX_RECEIPT_BYTES + 1))
        except (OSError, ValueError):
            report = None
        if code != 0 or not isinstance(report, dict):
            raise WorkerFailure(f"stage_child_failed:exit_{code}")
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _undeclared_writes(descriptor: Mapping[str, Any], runtime: WorkerRuntime) -> list[str]:
    """Every path under a path root that is not an input, not under the output root or declared scratch, and
    not a directory on the way to one of them."""
    outputs = descriptor["outputs"]
    trees = [outputs["output_root"], *(path.rstrip("/") for path in outputs["declared_scratch"])]
    kept = {item["materialize_at"] for item in descriptor["inputs"]}
    kept |= {str(parent) for path in (*trees, *kept) for parent in PurePosixPath(path).parents}
    found: list[str] = []
    for root in descriptor["limits"]["allowed_path_roots"]:
        for directory, folders, files in os.walk(runtime.local(root)):
            parent = "/" + os.path.relpath(directory, runtime.filesystem_root).strip("/")
            for name in (*folders, *files):
                path = f"{parent}/{name}"
                if path not in kept and not any(path == tree or path.startswith(tree + "/") for tree in trees):
                    found.append(path)
            folders[:] = [name for name in folders if f"{parent}/{name}" not in trees]
    return sorted(found)


def _integrity_failures(descriptor: Mapping[str, Any], runtime: WorkerRuntime) -> list[str]:
    """A changed input, or a write under a path root outside the declared outputs (plan 14 §6)."""
    changed = [f"input_changed:{safe_label(item['materialize_at'])}" for item in descriptor["inputs"]
               if not _unchanged(runtime.local(item["materialize_at"]), item)]
    written = [f"undeclared_write:{safe_label(path)}" for path in _undeclared_writes(descriptor, runtime)]
    return changed + written[:MAX_NAMED_PATHS]


def _bundle_members(path: Path) -> list[tuple[str, str]]:
    """``(member, digest)`` for each manifest entry that an input bundle's zip itself holds."""
    try:
        with zipfile.ZipFile(path) as bundle:
            names = set(bundle.namelist())
            entries = json.loads(bundle.read(BUNDLE_MANIFEST)).get("entries") if BUNDLE_MANIFEST in names else None
    except (OSError, ValueError, AttributeError, RuntimeError, zipfile.BadZipFile):
        return []
    return [(row["relative_path"], row["sha256"]) for row in (entries if isinstance(entries, list) else [])
            if isinstance(row, dict) and row.get("relative_path") in names and _DIGEST.fullmatch(str(row.get("sha256")))
            and all(part not in {"", ".", ".."} for part in str(row["relative_path"]).split("/"))]


def _host_known(descriptor: Mapping[str, Any], runtime: WorkerRuntime) -> dict[str, Any]:
    """An input's bytes by its digest, and an input bundle's members by the bundle's own manifest (plan 14 §6);
    landing re-verifies every member it extracts, so a false manifest fails closed on the host."""
    known: dict[str, Any] = {item["digest"]: {"input": item["digest"]} for item in descriptor["inputs"]}
    for item in descriptor["inputs"]:
        for member, digest in _bundle_members(runtime.local(item["materialize_at"])):
            known.setdefault(digest, {"input_member": {"input": item["digest"], "member": member}})
    return known


def _seal(attempt: _Attempt, runtime: WorkerRuntime, deadline: _Deadline) -> dict[str, Any]:
    """Upload ``blobs.tar`` (new bytes only) and ``index.json``; the receipt, which names them, comes after."""
    descriptor, limits = attempt.descriptor, attempt.descriptor["limits"]
    root = runtime.local(descriptor["outputs"]["output_root"])
    try:
        index = index_tree(root, host_known=_host_known(descriptor, runtime))
        index_data, size = index_bytes(index), blobs_tar_size(index)
    except RemoteCpuArchiveError as exc:
        raise WorkerFailure(f"seal_failed:{safe_label(str(exc))}") from None
    if index["paths_total"] > limits["max_output_paths"] or index["bytes_total"] > limits["max_output_bytes"]:
        raise WorkerFailure("output_exceeds_limits")
    if attempt.receipt_exists():  # another execution committed: its blobs and index are the ones it names
        raise _Committed("receipt_already_committed")
    written: dict[str, Any] = {}
    attempt.put("blobs.tar", lambda sink: written.update(write_blobs_tar(root, index, sink)), size=size,
                content_type="application/x-tar", timeout=deadline.timeout())
    deadline.check()
    attempt.put("index.json", index_data, timeout=deadline.timeout())
    deadline.check()
    return {"format": OUTPUT_FORMAT, "paths_total": index["paths_total"], "bytes_total": index["bytes_total"],
            "index": {"digest": "sha256:" + hashlib.sha256(index_data).hexdigest(), "size_bytes": len(index_data)},
            "archive": {"digest": written["digest"], "size_bytes": written["size_bytes"]},
            "host_known": index["host_known"]}


def _stage_outcome(attempt: _Attempt, runtime: WorkerRuntime, release: Path) -> tuple[str, Any, Any, list[str], list[str]]:
    """Fetch, run the stage child, check it, and seal a success: ``(status, result, output, codes, misses)``."""
    descriptor = attempt.descriptor
    with _phase(attempt, "fetch") as deadline:
        _materialize_inputs(attempt, runtime, deadline)
    handler = runtime.handlers.get(descriptor["stage"])
    if handler is None:
        raise WorkerFailure(f"stage_not_registered:{descriptor['stage']}")
    with _phase(attempt, "stage") as deadline:
        report = (runtime.run_stage or _run_stage_child)(runtime=runtime, descriptor=descriptor, release=release,
                                                         handler=handler, seconds=deadline.seconds)
    misses, failures = report.get("release_path_misses"), report.get("failures")
    if not (isinstance(misses, list) and isinstance(failures, list) and all(isinstance(item, str)
                                                                          for item in (*misses, *failures))):
        raise WorkerFailure("stage_report_invalid")
    result = report.get("result")
    # A release-path miss overrides any result the stage reached: it is never terminal (plan 14 §5, §10).
    codes = [*failures, *_integrity_failures(descriptor, runtime),
             *(RELEASE_PATH_MISSING.removeprefix(INFRASTRUCTURE_FAILED) + miss for miss in misses)]
    if codes:
        return "infrastructure_failed", result, None, codes, misses
    status = result.get("status") if isinstance(result, dict) else None
    if status == "blocked":
        return "blocked", result, None, [], []
    if status != STAGES[descriptor["stage"]]["success_status"]:
        raise WorkerFailure("stage_result_invalid")
    with _phase(attempt, "seal_upload") as deadline:
        return "succeeded", result, _seal(attempt, runtime, deadline), [], []


def execute_attempt(handoff: Mapping[str, Any], runtime: WorkerRuntime, *, shadowed: bool = False) -> int:
    """Run one attempt from bootstrap's handoff and commit its receipt last; exit 0 once it is up."""
    try:
        transport = validated_transport(handoff["transport"], runtime.environ)
        release, descriptor = Path(handoff["release_root"]), transport["descriptor"]
        if (handoff["schema_version"] != HANDOFF_SCHEMA_VERSION
                or handoff["execution_name"] != runtime.environ.get("CLOUD_RUN_EXECUTION")
                or release != runtime.local(RELEASE_PARENT) / descriptor["code"]["source_commit"]):
            raise WorkerFailure("remote_cpu_worker_handoff_invalid")
        attempt = _Attempt(runtime, transport, handoff["execution_name"],
                           started=runtime.clock() - float(handoff["elapsed_seconds"]),
                           sequence=int(handoff["heartbeat_sequence"]), fetched=int(handoff["bytes_fetched"]),
                           uploaded=int(handoff["bytes_uploaded"]), phases=handoff["phases"])
    except WorkerFailure as exc:
        return _refuse(runtime, "execute", exc.code)
    except (KeyError, TypeError, ValueError, AttributeError):
        return _refuse(runtime, "execute", "remote_cpu_worker_handoff_invalid")
    attempt.phase = "fetch"
    try:
        with _heartbeats(attempt):
            if shadowed:
                raise WorkerFailure("remote_cpu_worker_source_shadowed")
            status, result, output, codes, misses = _stage_outcome(attempt, runtime, release)
    except _Committed:
        runtime.log(_line("execute", "receipt_already_committed"))
        return 0
    except WorkerFailure as exc:
        return attempt.fail(exc.code, mode="execute")
    except Exception as exc:  # noqa: BLE001 - an unexpected error's message could carry anything; its type cannot
        return attempt.fail(f"remote_cpu_worker_raised:{type(exc).__name__}", mode="execute")
    return attempt.finish(status, mode="execute", result=result, output=output, codes=codes, misses=misses)


def _execute_main() -> int:
    """``execute``: read bootstrap's handoff from stdin and refuse unless this code is the release's."""
    runtime = WorkerRuntime(environ=os.environ, http=PresignedTransfers())
    try:
        handoff = json.loads(sys.stdin.buffer.read(MAX_HANDOFF_BYTES + 1))
        runtime.filesystem_root = Path(handoff["filesystem_root"])
        shadowed = not from_release(handoff["release_root"])
    except (ValueError, KeyError, TypeError):
        return _refuse(runtime, "execute", "remote_cpu_worker_handoff_invalid")
    return execute_attempt(handoff, runtime, shadowed=shadowed)


def bootstrap(argv: Sequence[str], runtime: WorkerRuntime) -> int:
    """The job's command: guard, read the pinned transport, fetch and extract the release, start ``execute``."""
    started = runtime.clock()
    try:
        execution_name = _guards(argv, runtime.environ)
        transport = read_transport(runtime)
    except WorkerFailure as exc:
        return _refuse(runtime, "bootstrap", exc.code)
    attempt = _Attempt(runtime, transport, execution_name, started=started)
    try:
        if attempt.receipt_exists():
            runtime.log(_line("bootstrap", "duplicate_execution"))
            return 0
    except WorkerFailure as exc:
        return _refuse(runtime, "bootstrap", exc.code)
    attempt.beat()
    try:
        _environment_gate(attempt)
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
    except Exception as exc:  # noqa: BLE001 - its message could carry anything; its type cannot
        return attempt.fail(f"remote_cpu_worker_raised:{type(exc).__name__}", mode="bootstrap")
    return 0 if code == 0 else attempt.fail(f"remote_cpu_worker_execute_failed:exit_{code}", mode="bootstrap")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(list(sys.argv[1:] if argv is None else argv))
    except Exception as exc:  # noqa: BLE001 - a traceback prints its message, which could carry a URL
        _log_stderr(_line("worker", "failed", type(exc).__name__))
        return EXIT_REFUSED


def _main(arguments: list[str]) -> int:
    if arguments[:1] == ["bootstrap"]:
        os.umask(0o077)  # as the host unit does; execute and the stage child inherit it
        return bootstrap(arguments, WorkerRuntime(environ=os.environ, http=PresignedTransfers()))
    if arguments == ["execute"]:
        return _execute_main()
    if arguments == ["stage"]:
        return stage_main()
    if arguments == ["environment"]:  # the host census's record, measured here (plan 14 §5)
        print(json.dumps(environment_record(), sort_keys=True))
        return 0
    return _refuse(WorkerRuntime(environ=os.environ, http=None), "worker", "remote_cpu_worker_argv_invalid")


if __name__ == "__main__":
    raise SystemExit(main())
