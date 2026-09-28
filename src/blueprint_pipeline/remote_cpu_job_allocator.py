"""The paid seam for remote CPU jobs on Cloud Run (plan 14 §2, §4, §8, §10, §11).

``python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job`` is the only creator of
remote CPU executions.  Every attempt is admitted through the shared fail-closed chokepoint
(``build_paid_lane_admission``, then its allocation binding, then
``require_paid_resource_admission``) against the owner's standing authority, a worst-case spend
ledger and the live-execution cap, and its authority is consumed exactly once before anything is
created.  The allocator never mints an authority, and nothing here writes a URL or a credential to
host disk or a log.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import remote_cpu_job_lease as leases
from .cloud_run_jobs_client import (
    ATTEMPT_VARIABLE, CLOUD_RUN_CPU_JOB_RESOURCE_CLASS, DESCRIPTOR_VARIABLE, TRANSPORT_GENERATION_VARIABLE,
    TRANSPORT_OBJECT_VARIABLE, CloudRunAmbiguousResponse, CloudRunJobsClient, CloudRunJobsError, GcsTransportBucket,
    allocation_binding, delete_transport_object, env_value, execution_resource_name, job_definition_blockers,
    job_resource_name, load_dispatcher_credentials, run_target, transport_bucket_sentinel,
)
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION, PaidResourceAdmissionBlocked, PaidResourceAdmissionGrant,
    build_paid_lane_admission, require_paid_resource_admission, require_paid_resource_admission_grant,
)
from .remote_cpu_environment import DIGESTED_FIELDS, environment_record
from .remote_cpu_job_contract import (
    CONFIG_SCHEMA_VERSION, MAX_ATTEMPTS_CAP, MAX_HEARTBEAT_BYTES, MAX_RECEIPT_BYTES, STAGES, TRANSPORT_SCHEMA_VERSION,
    RemoteCpuContractError, _is_count, _text, build_descriptor, execution_name_of, record_bytes, validate_descriptor,
    validate_receipt, worker_identity_for,
)
from .remote_cpu_job_records import compute_zero_proven, fsync_directory, teardown_record, write_remote_cpu_record
from .spend_authority_consumption_root import (
    SpendAuthorityRootError, authorizations_root, prepare_consumption_root, spend_authority_root,
)
from .task_evaluation_configured_scene_object_store import (
    DEFAULT_KEY_PREFIX, TaskEvaluationConfiguredSceneObjectStoreError, delete_remote_cpu_staging_versions,
    presign_remote_cpu_get, presign_remote_cpu_put, publish_configured_scene_stream, read_remote_cpu_staging_object,
    remote_cpu_object_store, remote_cpu_object_store_sentinel,
)

RESOURCE_CLASS = CLOUD_RUN_CPU_JOB_RESOURCE_CLASS
ACTIONS = ("dispatch", "reconcile", "cancel", "sweep", "preflight")
PROBE_STAGE = "environment_probe"
CONFIG_ENV = "BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/blueprint/remote-cpu-workers.json"
AUTHORITY_FILENAME = "remote-cpu-standing-authorization.v1.json"
AUTHORITY_SCHEMA_VERSION = "remote_cpu_standing_authorization.v1"
CONSUMPTION_SCHEMA_VERSION = "remote_cpu_spend_consumption.v1"
SETTLEMENT_SCHEMA_VERSION = "remote_cpu_spend_settlement.v1"
RESULT_SCHEMA_VERSION = "remote_cpu_job_result.v1"
ENVIRONMENT_SCHEMA_VERSION = "remote_cpu_worker_environment.v1"
PROBE_REQUEST_SCHEMA_VERSION = "remote_cpu_environment_probe_request.v1"
PROBE_ROOT = "/var/lib/blueprint/remote-cpu-probes"
RECONCILE_AFTER_SECONDS = 120  # past a :run request's own timeout, so reconcile never races a dispatch
CANCEL_GRACE_SECONDS = 300
SETTLED_DIRECTORY = "remote-cpu-settled"
LEDGER_LOCK = "remote-cpu.lock"
DAY_SECONDS = 86400
GIB = 1024**3
# The worker's write authority: one presigned PUT per staging object (plan 14 §4, §6).
STAGING_OBJECTS = ("blobs.tar", "index.json", "receipt.json", "heartbeat.json")
# Plan 14 §3/§10: fetch 300 s + stage 900 s + seal/upload 420 s + a 180 s margin fill the 1800 s task.
STAGE_LIMITS: Mapping[str, Any] = {
    "phase_seconds": {"fetch": 300, "stage": 900, "seal_upload": 420}, "start_allowance_seconds": 600,
    "heartbeat_interval_seconds": 30, "heartbeat_stale_seconds": 180, "max_input_bytes": 6 * GIB,
    "max_output_bytes": 4 * GIB, "max_output_paths": 20000, "allowed_path_roots": ["/var/lib/blueprint/"],
}
_CONFIG_KEYS = frozenset({"schema_version", "project", "region", "transport_bucket", "stages", "rate_table",
                          "max_live_executions", "max_attempts", "config_digest"})
_STAGE_KEYS = frozenset({"job", "image", "vcpu", "memory_bytes", "ephemeral_bytes", "task_timeout_seconds"})
_RATES = ("usd_per_vcpu_second", "usd_per_gib_second", "usd_per_egress_gib")
_AUTHORITY_KEYS = frozenset({"schema_version", "stages", "max_executions", "max_attempt_usd", "max_daily_usd",
                             "max_total_usd", "expires_at_epoch", "authorized_by", "authorized_on",
                             "authorization_reference", "authorization_digest"})
_IMAGE = re.compile(r"[a-z0-9][a-z0-9.-]*(?::[0-9]+)?/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}")
_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_JOB = re.compile(r"blueprint-remote-cpu-[a-z0-9-]{0,40}[a-z0-9]")
_TYPED = re.compile(r"[a-z0-9_]+(?::[A-Za-z0-9_.-]+)*")
_MAX_RECORD_BYTES = 256 * 1024


class RemoteCpuAllocatorError(RuntimeError):
    """A typed refusal of the remote CPU seam; ``code`` is the blocker."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def _amount(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _micros(value: float) -> int:
    return round(float(value) * 1_000_000)


def _read_private_json(path: str | Path, *, forbidden_mode: int) -> dict[str, Any]:
    """A regular, non-symlinked JSON object whose mode has none of ``forbidden_mode``'s bits."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise RemoteCpuAllocatorError("missing") from None
    except OSError:
        raise RemoteCpuAllocatorError("unsafe") from None
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_mode & forbidden_mode:
            raise RemoteCpuAllocatorError("unsafe")
        payload = stream.read(_MAX_RECORD_BYTES + 1)
    try:
        value = json.loads(payload)
    except ValueError:
        value = None
    if not isinstance(value, dict) or len(payload) > _MAX_RECORD_BYTES:
        raise RemoteCpuAllocatorError("unreadable")
    return value


def _write_file(path: Path, payload: bytes, *, mode: int, exclusive: bool) -> bool:
    """Write fsynced bytes beside ``path``, then link them in (``exclusive``: ``False`` if it exists) or replace it."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(temporary, flags, mode), "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not exclusive:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)
    return True


def _create_once(path: Path, value: Mapping[str, Any]) -> bool:
    """A spend record: O_EXCL plus a link, fsynced and 0600; ``False`` when it already exists."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return _write_file(path, payload.encode("utf-8"), mode=0o600, exclusive=True)


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically replace a door-readable record; the record guard refuses URL- or credential-shaped content."""
    _write_file(path, record_bytes(result), mode=0o640, exclusive=False)


def _stage_entry_valid(stage: str, entry: Any) -> bool:
    return (stage in STAGES and stage != PROBE_STAGE and isinstance(entry, Mapping) and set(entry) == _STAGE_KEYS
            and isinstance(entry["job"], str) and _JOB.fullmatch(entry["job"]) is not None
            and isinstance(entry["image"], str) and _IMAGE.fullmatch(entry["image"]) is not None
            and all(_is_count(entry[name], 1) for name in ("vcpu", "memory_bytes", "ephemeral_bytes",
                                                         "task_timeout_seconds"))
            and entry["task_timeout_seconds"] <= 3600 and entry["memory_bytes"] <= 32 * GIB)


def config_blockers(config: Mapping[str, Any]) -> list[str]:
    """Every way a ``remote_cpu_workers_config.v1`` is unusable; its region must be a US one."""
    stages = config.get("stages") if isinstance(config.get("stages"), Mapping) else {}
    rates = config.get("rate_table") if isinstance(config.get("rate_table"), Mapping) else {}
    region = str(config.get("region") or "")
    failed = {
        "keys": set(config) != _CONFIG_KEYS,
        "schema_version": config.get("schema_version") != CONFIG_SCHEMA_VERSION,
        "config_digest": config.get("config_digest") != canonical_digest(config, digest_field="config_digest"),
        "project": not isinstance(config.get("project"), str) or _PROJECT.fullmatch(config["project"]) is None,
        "region": region.startswith("us-") and re.fullmatch(r"us-[a-z]+[0-9]+", region) is None,
        "transport_bucket": _BUCKET.fullmatch(str(config.get("transport_bucket") or "")) is None,
        "stages": not stages or not all(_stage_entry_valid(stage, entry) for stage, entry in stages.items()),
        "rate_table": set(rates) != {"source", "observed_on", *_RATES} or not all(_amount(rates[n]) for n in _RATES)
        or not _text(rates["source"]) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", str(rates["observed_on"])),
        "max_live_executions": not _is_count(config.get("max_live_executions"), 1),
        "max_attempts": not _is_count(config.get("max_attempts"), 1) or config["max_attempts"] > MAX_ATTEMPTS_CAP,
    }
    blockers = [f"remote_cpu_config_invalid:{name}" for name, failure in failed.items() if failure]
    if not region.startswith("us-"):
        blockers.append("remote_cpu_config_region_not_us")
    return sorted(blockers)


def load_remote_cpu_config(path: str | Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """The host's ``remote_cpu_workers_config.v1`` (``BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG``, 0640)."""
    location = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    try:
        config = _read_private_json(location, forbidden_mode=0o027)
    except RemoteCpuAllocatorError as exc:
        return {}, [f"remote_cpu_config_invalid:{exc.code}"]
    return config, config_blockers(config)


def load_standing_authority(*, stage: str, now: float,
                            path: str | Path | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    """The owner's ``remote_cpu_standing_authorization.v1``, provisioned 0600; the allocator never mints one."""
    location = Path(path) if path else authorizations_root() / AUTHORITY_FILENAME
    try:
        authority = _read_private_json(location, forbidden_mode=0o077)
    except RemoteCpuAllocatorError as exc:
        return None, ["remote_cpu_standing_authority_missing" if exc.code == "missing"
                      else f"remote_cpu_standing_authority_invalid:{exc.code}"]
    failed = {
        "keys": set(authority) != _AUTHORITY_KEYS,
        "schema_version": authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION,
        "authorization_digest": authority.get("authorization_digest") != canonical_digest(
            authority, digest_field="authorization_digest"),
        "stages": not isinstance(authority.get("stages"), list) or not all(map(_text, authority["stages"])),
        "caps": not all(_amount(authority.get(name)) for name in ("max_attempt_usd", "max_daily_usd", "max_total_usd")),
        "max_executions": not _is_count(authority.get("max_executions"), 1),
        "expires_at_epoch": not _amount(authority.get("expires_at_epoch")),
        "authorized": not all(_text(authority.get(name)) for name in ("authorized_by", "authorized_on",
                                                                     "authorization_reference")),
    }
    reasons = [name for name, failure in failed.items() if failure]
    if reasons:
        return None, [f"remote_cpu_standing_authority_invalid:{reasons[0]}"]
    blockers = []
    if float(now) >= authority["expires_at_epoch"]:
        blockers.append("remote_cpu_standing_authority_expired")
    if stage not in authority["stages"]:
        blockers.append("remote_cpu_standing_authority_stage_not_covered")
    return authority, blockers


def worst_case_usd(*, limits: Mapping[str, Any], rate_table: Mapping[str, Any]) -> float:
    """Task timeout x (vCPU x rate + GiB x rate), plus every output byte as egress; rounded up to a micro-dollar."""
    compute = limits["task_timeout_seconds"] * (limits["vcpu"] * rate_table["usd_per_vcpu_second"]
                                                + limits["memory_bytes"] / GIB * rate_table["usd_per_gib_second"])
    egress = limits["max_output_bytes"] / GIB * rate_table["usd_per_egress_gib"]
    return math.ceil((compute + egress) * 1_000_000 - 1e-6) / 1_000_000


def stage_limits(config: Mapping[str, Any], stage: str, *, allowed_cpu_classes: tuple[str, ...] = ()) -> dict[str, Any]:
    """A descriptor's limits: the stage job's resources, plan 14's phase budget and the config's attempts."""
    entry = config["stages"][stage]
    return {**{name: entry[name] for name in ("vcpu", "memory_bytes", "ephemeral_bytes", "task_timeout_seconds")},
            **json.loads(json.dumps(STAGE_LIMITS)), "max_attempts": config["max_attempts"],
            "allowed_cpu_classes": sorted(allowed_cpu_classes)}


def _attempt_key(attempt_id: str) -> str:
    return hashlib.sha256(str(attempt_id).encode("utf-8")).hexdigest()


def _consumption_path(attempt_id: str) -> Path:
    return prepare_consumption_root() / f"remote-cpu-{_attempt_key(attempt_id)}.json"


def _settled_root() -> Path:
    root = spend_authority_root() / SETTLED_DIRECTORY
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


@contextmanager
def spend_ledger_lock() -> Iterator[None]:
    """Serialize admission, consumption and settlement under ``remote-cpu.lock``."""
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(_settled_root() / LEDGER_LOCK, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _sealed(path: Path, schema_version: str, digest_field: str) -> dict[str, Any]:
    record = _read_private_json(path, forbidden_mode=0o077)
    if record.get("schema_version") != schema_version or record.get(digest_field) != canonical_digest(
            record, digest_field=digest_field):
        raise RemoteCpuAllocatorError("unreadable")
    return record


def spend_ledger(authority_digest: str) -> list[dict[str, Any]]:
    """Every attempt consumed under this authority, at its settled estimate or, until settled, its worst case."""
    settled, rows = _settled_root(), []
    for path in sorted(prepare_consumption_root().glob("remote-cpu-*.json")):
        consumption = _sealed(path, CONSUMPTION_SCHEMA_VERSION, "consumption_digest")
        if consumption.get("standing_authority_digest") != authority_digest:
            continue
        settlement = settled / f"{_attempt_key(consumption['attempt_id'])}.json"
        usd = consumption["worst_case_usd"]
        if settlement.exists():
            usd = _sealed(settlement, SETTLEMENT_SCHEMA_VERSION, "settlement_digest")["settled_usd"]
        if not isinstance(usd, (int, float)) or not isinstance(consumption["consumed_at_epoch"], (int, float)):
            raise RemoteCpuAllocatorError("unreadable")
        rows.append({"attempt_id": consumption["attempt_id"], "consumed_at_epoch": consumption["consumed_at_epoch"],
                     "usd": float(usd)})
    return rows


def spend_ledger_blockers(*, authority: Mapping[str, Any], worst_case_usd: float, now: float) -> list[str]:
    """The standing authority's caps against the ledger plus this attempt's worst case (plan 14 §8)."""
    try:
        rows = spend_ledger(authority["authorization_digest"])
    except (RemoteCpuAllocatorError, OSError, SpendAuthorityRootError, KeyError, TypeError, ValueError):
        return ["remote_cpu_spend_ledger_unreadable"]
    day = sum(row["usd"] for row in rows if row["consumed_at_epoch"] > float(now) - DAY_SECONDS)
    total = sum(row["usd"] for row in rows)
    failed = {
        "remote_cpu_attempt_cap_exceeded": _micros(worst_case_usd) > _micros(authority["max_attempt_usd"]),
        "remote_cpu_daily_cap_exceeded": _micros(day + worst_case_usd) > _micros(authority["max_daily_usd"]),
        "remote_cpu_total_cap_exceeded": _micros(total + worst_case_usd) > _micros(authority["max_total_usd"]),
        "remote_cpu_execution_cap_exceeded": len(rows) + 1 > authority["max_executions"],
    }
    return sorted(name for name, failure in failed.items() if failure)


def consume_remote_cpu_authority_once(*, descriptor: Mapping[str, Any], authority: Mapping[str, Any],
                                      worst_case_usd: float, binding_digest: str, now: float) -> dict[str, Any]:
    """Consume the standing authority for one attempt, exactly once, recording its worst case."""
    record: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA_VERSION, "attempt_id": descriptor["attempt_id"],
        "job_id": descriptor["job_id"], "stage": descriptor["stage"], "descriptor_digest": descriptor["descriptor_digest"],
        "standing_authority_digest": authority["authorization_digest"], "allocation_binding_digest": binding_digest,
        "worst_case_usd": float(worst_case_usd), "consumed_at_epoch": float(now), "maximum_executions": 1,
        "consumption_digest": "",
    }
    record["consumption_digest"] = canonical_digest(record, digest_field="consumption_digest")
    try:
        created = _create_once(_consumption_path(descriptor["attempt_id"]), record)
    except (OSError, SpendAuthorityRootError):
        return {"status": "blocked", "blockers": ["remote_cpu_authority_consumption_failed"]}
    if not created:
        return {"status": "blocked", "blockers": ["remote_cpu_authority_already_consumed"]}
    return {"status": "consumed", "consumption_digest": record["consumption_digest"]}


def settle_remote_cpu_attempt(*, descriptor: Mapping[str, Any], teardown_digest: str, settled_usd: float,
                              basis: str, now: float) -> dict[str, Any]:
    """Replace an attempt's worst case in the ledger with its estimate once its teardown is sealed."""
    key = _attempt_key(descriptor["attempt_id"])
    consumption = _sealed(_consumption_path(descriptor["attempt_id"]), CONSUMPTION_SCHEMA_VERSION, "consumption_digest")
    record: dict[str, Any] = {
        "schema_version": SETTLEMENT_SCHEMA_VERSION, "attempt_id": descriptor["attempt_id"],
        "standing_authority_digest": consumption["standing_authority_digest"],
        "worst_case_usd": consumption["worst_case_usd"], "settled_usd": round(float(settled_usd), 6), "basis": basis,
        "teardown_digest": teardown_digest, "settled_at_epoch": float(now), "settlement_digest": "",
    }
    record["settlement_digest"] = canonical_digest(record, digest_field="settlement_digest")
    with spend_ledger_lock():
        created = _create_once(_settled_root() / f"{key}.json", record)
    return {"status": "settled" if created else "already_settled", "settled_usd": record["settled_usd"]}


def environment_blockers(root: Path, descriptor: Mapping[str, Any], *, image: str) -> list[str]:
    """Dispatch needs the worker environment the preflight probe recorded for this image, equal to the host's."""
    try:
        record = _read_private_json(Path(root) / "environment" / f"{descriptor['stage']}.json", forbidden_mode=0o022)
    except RemoteCpuAllocatorError:
        return ["remote_cpu_environment_unrecorded"]
    if record.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION or record.get("image") != image:
        return ["remote_cpu_environment_unrecorded"]
    if record.get("environment_digest") != descriptor["code"]["environment_digest"]:
        return ["remote_cpu_environment_mismatch"]
    return []


def admit_remote_cpu_job(*, blockers: list[str], binding: Mapping[str, Any],
                         execute: bool) -> tuple[dict[str, Any], PaidResourceAdmissionGrant | None]:
    """Admit one attempt as ``paid_resource_allocator.py`` admits a lane: build, then bind, then require."""
    admission = build_paid_lane_admission(resource_class=RESOURCE_CLASS, blockers=sorted(set(blockers)))
    admission.update({"allocation_binding": dict(binding), "allocation_binding_digest": canonical_digest(binding)})
    if not execute:
        return admission, None
    try:
        grant = require_paid_resource_admission(admission, resource_class=RESOURCE_CLASS,
                                                expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    except PaidResourceAdmissionBlocked:
        return admission, None
    return admission, grant


def mint_transport(*, grant: PaidResourceAdmissionGrant | None, binding_digest: str, descriptor: Mapping[str, Any],
                   bucket: Any, object_store: tuple[Any, str, str], now: float) -> dict[str, Any]:
    """Presign an admitted attempt's transport and write it create-if-absent (plan 14 §4): GETs expire when
    the fetch window closes, PUTs at the hard deadline.  It exists only in memory and in its GCS object;
    the caller gets its name, generation and both expiries, for the lease, before the attempt dispatches."""
    require_paid_resource_admission_grant(grant, resource_class=RESOURCE_CLASS, allocation_binding_digest=binding_digest,
                                          require_allocation_binding=True)
    client, b2_bucket, limits = object_store[0], object_store[1], descriptor["limits"]
    write_expiry = leases.hard_deadline_epoch(now, limits)
    read_expiry = (float(now) + limits["start_allowance_seconds"] + limits["phase_seconds"]["fetch"]
                   + leases.HARD_DEADLINE_GRACE_SECONDS)
    reads, writes = math.floor(read_expiry - float(now)), math.floor(write_expiry - float(now))
    staging, archive = descriptor["outputs"]["staging_prefix"], descriptor["code"]["source_archive"]

    def get(uri: str) -> str:
        return presign_remote_cpu_get(uri=uri, expires_in_seconds=reads, client=client, bucket=b2_bucket)

    transport = {
        "schema_version": TRANSPORT_SCHEMA_VERSION, "descriptor": dict(descriptor),
        "inputs": [{"materialize_at": item["materialize_at"], "digest": item["digest"],
                    "size_bytes": item["size_bytes"], "url": get(item["uri"])} for item in descriptor["inputs"]],
        "source_archive": {"digest": archive["digest"], "size_bytes": archive["size_bytes"], "url": get(archive["uri"])},
        "receipt_url": get(staging + "receipt.json"),
        "outputs": {name: presign_remote_cpu_put(staging_uri=staging + name, expires_in_seconds=writes, client=client,
                                                 bucket=b2_bucket) for name in STAGING_OBJECTS},
        "read_urls_expire_at_epoch": read_expiry, "write_urls_expire_at_epoch": write_expiry,
    }
    name = f"transport/{descriptor['job_id']}/{descriptor['attempt_id']}-{secrets.token_hex(16)}.json"
    payload = json.dumps(transport, sort_keys=True, separators=(",", ":")).encode("utf-8")
    generation = bucket.create(name, payload, if_generation_match=0)
    return {"transport_object": f"gs://{bucket.name}/{name}", "transport_generation": int(generation),
            "write_urls_expire_at_epoch": write_expiry, "read_urls_expire_at_epoch": read_expiry}


@dataclass
class RemoteCpuRuntime:
    """What the seam touches beyond this host; production builds each service lazily from its credential.
    ``stage_release_source`` (plan 14 task 3.1) returns the release commit and its CAS source archive;
    until it is wired, the preflight probe refuses before any mutation."""

    config_path: str | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    cloud_run: Any = None
    transport_bucket: Any = None
    object_store: tuple[Any, str, str] | None = None
    stage_release_source: Callable[[], Mapping[str, Any]] | None = None
    host_environment: Callable[[], Mapping[str, Any]] = environment_record
    presigned_put: Callable[[str, bytes], int] | None = None
    poll_seconds: float = 15.0


@dataclass
class _Action:
    runtime: RemoteCpuRuntime
    config: dict[str, Any]
    root: Path
    execute: bool
    now: float
    authority: dict[str, Any] | None = None
    blockers: list[str] = field(default_factory=list)


def _connect(runtime: RemoteCpuRuntime, config: Mapping[str, Any]) -> list[str]:
    """The dispatcher's Cloud Run client and transport bucket, and the B2 store, which must be in the US."""
    try:
        if runtime.cloud_run is None or runtime.transport_bucket is None:
            credentials = load_dispatcher_credentials()
            if runtime.cloud_run is None:
                runtime.cloud_run = CloudRunJobsClient(credentials=credentials)
            if runtime.transport_bucket is None:
                runtime.transport_bucket = GcsTransportBucket(config["transport_bucket"], credentials=credentials,
                                                              project=config["project"])
        if runtime.object_store is None:
            runtime.object_store = remote_cpu_object_store()
    except CloudRunJobsError as exc:
        return [exc.code]
    except TaskEvaluationConfiguredSceneObjectStoreError as exc:
        code = str(exc)
        return [code if _TYPED.fullmatch(code) else "remote_cpu_object_store_unavailable"]
    except Exception as exc:  # noqa: BLE001 - client construction is outside this seam; its cause stays typed
        return [f"remote_cpu_dispatcher_unavailable:{type(exc).__name__}"]
    return [] if str(runtime.object_store[2] or "").startswith("us-") else ["remote_cpu_object_store_region_not_us"]


def _lease(root: Path, job_id: str) -> dict[str, Any] | None:
    try:
        lease = _read_private_json(Path(root) / "leases" / f"{job_id}.json", forbidden_mode=0o022)
    except RemoteCpuAllocatorError as exc:
        if exc.code == "missing":
            return None
        raise RemoteCpuAllocatorError("remote_cpu_lease_unreadable") from None
    if lease.get("lease_digest") != canonical_digest(lease, digest_field="lease_digest"):
        raise RemoteCpuAllocatorError("remote_cpu_lease_unreadable")
    return lease


def _probe_config(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """The config as a probe of ``stage`` sees it: its one stage is the probe, on that stage's job."""
    derived = {**config, "stages": {PROBE_STAGE: dict(config["stages"][stage])}, "config_digest": ""}
    derived["config_digest"] = canonical_digest(derived, digest_field="config_digest")
    return derived


def _config_for(config: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """A descriptor's config, and the configured stage whose job it runs on."""
    if raw.get("stage") != PROBE_STAGE:
        return dict(config), str(raw.get("stage"))
    execution = raw.get("execution") if isinstance(raw.get("execution"), Mapping) else {}
    stage = next((name for name, entry in config["stages"].items() if entry["job"] == execution.get("job")), None)
    if stage is None:
        raise RemoteCpuAllocatorError("remote_cpu_probe_stage_unknown")
    return _probe_config(config, stage), stage


def _dispatch(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Admit, lease, consume and mint for one attempt under ``remote-cpu.lock``; then start it once."""
    job_id, attempt_id = descriptor["job_id"], descriptor["attempt_id"]
    lease = _lease(action.root, job_id)
    if lease is not None and lease["attempt_id"] == attempt_id and lease["state"] not in ("claimed", "awaiting_capacity"):
        return {"status": "already_dispatched", "blockers": ["remote_cpu_attempt_already_dispatched"],
                "job_id": job_id, "attempt_id": attempt_id, "lease_state": lease["state"]}
    runtime, target, entry = action.runtime, run_target(descriptor), action.config["stages"][descriptor["stage"]]
    blockers = list(action.blockers)
    worst = worst_case_usd(limits=descriptor["limits"], rate_table=action.config["rate_table"])
    if _micros(worst) != _micros(descriptor["spend"]["worst_case_usd"]):
        blockers.append("remote_cpu_descriptor_spend_mismatch")
    if descriptor["stage"] != PROBE_STAGE:
        blockers += environment_blockers(action.root, descriptor, image=entry["image"])
    etag = ""
    try:
        job = runtime.cloud_run.get_job(target["job"])
        etag = str(job.get("etag") or "")
        blockers += job_definition_blockers(job, image=entry["image"], timeout_seconds=target["timeout_seconds"])
    except CloudRunJobsError as exc:
        blockers.append(f"remote_cpu_job_unavailable:{exc.code}")
    binding = allocation_binding(descriptor_digest=descriptor["descriptor_digest"], attempt_id=attempt_id,
                                 job=target["job"], etag=etag)
    with spend_ledger_lock():
        if _consumption_path(attempt_id).exists():
            blockers.append("remote_cpu_authority_already_consumed")
        if action.authority is not None:
            blockers += spend_ledger_blockers(authority=action.authority, worst_case_usd=worst, now=action.now)
        census = leases.slot_census(action.root)
        if census["slots_in_use"] >= action.config["max_live_executions"]:
            blockers.append("remote_cpu_live_execution_cap_reached")
        admission, grant = admit_remote_cpu_job(blockers=blockers, binding=binding, execute=action.execute)
        view = ("schema_version", "status", "resource_class", "blockers", "allocation_binding", "allocation_binding_digest")
        result = {"admission": {name: admission[name] for name in view}, "blockers": admission["blockers"], "job_id": job_id,
                  "attempt_id": attempt_id, "slots_in_use": census["slots_in_use"],
                  "unreadable_leases": census["unreadable"]}
        if admission["blockers"] == ["remote_cpu_live_execution_cap_reached"]:
            # A full slot table is a wait, never a fallback: the attempt keeps its claim.
            if action.execute:
                lease = leases.claim_handoff(action.root, descriptor=descriptor, config=action.config, now=action.now)
                if lease["state"] == "claimed":
                    leases.transition(action.root, job_id, attempt_id=attempt_id, to_state="awaiting_capacity",
                                      now=action.now)
            return {**result, "status": "awaiting_capacity"}
        if grant is None:
            return {**result, "status": "blocked" if admission["blockers"] else "dry_run_ready"}
        leases.claim_handoff(action.root, descriptor=descriptor, config=action.config, now=action.now)
        consumption = consume_remote_cpu_authority_once(
            descriptor=descriptor, authority=action.authority, worst_case_usd=worst,
            binding_digest=admission["allocation_binding_digest"], now=action.now)
        if consumption["status"] != "consumed":
            return {**result, "status": "blocked", "blockers": consumption["blockers"]}
        try:
            transport = mint_transport(grant=grant, binding_digest=admission["allocation_binding_digest"],
                                       descriptor=descriptor, bucket=runtime.transport_bucket,
                                       object_store=runtime.object_store, now=action.now)
        except Exception as exc:  # noqa: BLE001 - nothing ran; the consumed attempt now falls back
            return {**result, "status": "blocked", "blockers": [f"remote_cpu_transport_mint_failed:{type(exc).__name__}"]}
        try:
            leases.transition(action.root, job_id, attempt_id=attempt_id, to_state="dispatching", now=action.now,
                              updates=transport)
        except RemoteCpuContractError:
            with suppress(Exception):  # a transport that never reached its lease; lifecycle is the backstop
                delete_transport_object(runtime.transport_bucket, transport["transport_object"],
                                        transport["transport_generation"])
            raise
    result.update(transport_object=transport["transport_object"], transport_generation=transport["transport_generation"])
    return _start(action, descriptor, grant=grant, etag=etag, result=result)


def _start(action: _Action, descriptor: Mapping[str, Any], *, grant: PaidResourceAdmissionGrant, etag: str,
           result: dict[str, Any]) -> dict[str, Any]:
    """Run the admitted attempt once.  A probe first proves its sentinels and ``validate_only``; an
    ambiguous response is reconciled from a complete listing and never re-issued."""
    runtime, target, attempt_id = action.runtime, run_target(descriptor), descriptor["attempt_id"]
    overrides = {ATTEMPT_VARIABLE: attempt_id, DESCRIPTOR_VARIABLE: descriptor["descriptor_digest"],
                 TRANSPORT_OBJECT_VARIABLE: result["transport_object"],
                 TRANSPORT_GENERATION_VARIABLE: str(result["transport_generation"])}
    refusal, names = None, []
    if descriptor["stage"] == PROBE_STAGE:
        store = runtime.object_store
        sentinels = result["sentinels"] = {
            "transport": transport_bucket_sentinel(
                runtime.transport_bucket, f"transport/{descriptor['job_id']}/{attempt_id}-sentinel.json"),
            "object_store": remote_cpu_object_store_sentinel(
                staging_prefix=descriptor["outputs"]["staging_prefix"], attempt_id=attempt_id, client=store[0],
                bucket=store[1], put=runtime.presigned_put)}
        try:
            runtime.cloud_run.run_job(grant=grant, target=target, etag=etag, overrides=overrides, validate_only=True)
            sentinels["validate_only"] = {"status": "passed", "blockers": []}
        except CloudRunJobsError as exc:
            sentinels["validate_only"] = {"status": "blocked", "blockers": [exc.code]}
        failed = sorted(name for name, sentinel in sentinels.items() if sentinel["status"] != "passed")
        refusal = f"remote_cpu_preflight_sentinel_failed:{'.'.join(failed)}" if failed else None
    if refusal is None:
        try:
            operation = runtime.cloud_run.run_job(grant=grant, target=target, etag=etag, overrides=overrides)
            names = [str((operation.get("metadata") or {}).get("name") or "")]
        except CloudRunAmbiguousResponse:
            found = result["reconciled"] = reconcile_ambiguous_dispatch(action, descriptor)
            if found["status"] == "unresolved":
                return {**result, "status": "ambiguous_dispatch_unresolved", "blockers": found["blockers"]}
            names, refusal = found["executions"], None if found["executions"] else "remote_cpu_dispatch_lost"
        except CloudRunJobsError as exc:
            refusal = f"remote_cpu_dispatch_refused:{exc.code}"
    if names:
        return {**result, **_record_dispatched(action, descriptor, names[0])}
    torn = _teardown(action, descriptor, outcome=refusal, terminal="abandoned_dispatch",
                     wait=descriptor["stage"] == PROBE_STAGE)
    return {**result, **torn, "blockers": sorted({refusal, *torn["blockers"]})}


def _record_dispatched(action: _Action, descriptor: Mapping[str, Any], execution: str) -> dict[str, Any]:
    identity = worker_identity_for(descriptor["execution"], execution.rsplit("/", 1)[-1])
    leases.transition(action.root, descriptor["job_id"], attempt_id=descriptor["attempt_id"], to_state="dispatched",
                      now=float(action.runtime.clock()), updates={"worker_identity": identity})
    return {"status": "dispatched", "worker_identity": identity, "blockers": []}


def reconcile_ambiguous_dispatch(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Every execution carrying this attempt, from a listing followed to its last page (plan 14 §2)."""
    try:
        rows, pages = action.runtime.cloud_run.list_all_executions(run_target(descriptor)["job"])
    except CloudRunJobsError as exc:
        return {"status": "unresolved", "executions": [],
                "blockers": sorted({"remote_cpu_ambiguous_dispatch_unresolved", exc.code})}
    mine = sorted((row for row in rows if env_value(row, ATTEMPT_VARIABLE) == descriptor["attempt_id"]),
                  key=lambda row: str(row.get("createTime") or ""))
    return {"status": "found" if mine else "absent", "executions": [str(row.get("name") or "") for row in mine],
            "listing_pages": pages, "blockers": []}


def prove_compute_zero(action: _Action, lease: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Compute-zero evidence (plan 14 §11): the execution terminal, a complete listing with nothing of this
    attempt unfinished, and only then the transport deleted and absent at its generation."""
    cloud_run, job, identity = action.runtime.cloud_run, run_target(descriptor)["job"], lease["worker_identity"]
    execution = cloud_run.get_execution(execution_resource_name(job, execution_name_of(identity))) if identity else {}
    rows, pages = cloud_run.list_all_executions(job)
    mine = [row for row in rows if env_value(row, ATTEMPT_VARIABLE) == descriptor["attempt_id"]]
    unfinished = sum(1 for row in mine if not row.get("completionTime"))
    completed = bool(execution.get("completionTime")) and not execution.get("runningCount")
    absent = unfinished == 0 and (completed or not identity) and delete_transport_object(
        action.runtime.transport_bucket, lease["transport_object"], lease["transport_generation"])
    return {"execution_completed": completed, "running_count": int(execution.get("runningCount") or 0),
            "listing_complete": True, "listing_pages": pages, "executions_for_attempt": len(mine),
            "unfinished_executions_for_attempt": unfinished, "transport_object": lease["transport_object"],
            "transport_generation": lease["transport_generation"], "transport_deleted": absent,
            "transport_absent_at_generation": absent}


def prove_provider_zero(action: _Action, lease: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Provider-zero evidence: every staging version deleted and none listed, and the recorded URL expiries."""
    store = action.runtime.object_store
    deleted = delete_remote_cpu_staging_versions(staging_prefix=descriptor["outputs"]["staging_prefix"],
                                                 client=store[0], bucket=store[1])
    return {"staging_versions_deleted": deleted["versions_deleted"],
            "staging_versions_remaining": deleted["versions_remaining"],
            "staging_listing_complete": deleted["listing_complete"],
            **{name: lease[name] for name in ("write_urls_expire_at_epoch", "read_urls_expire_at_epoch")}}


def _epoch(text: Any) -> float | None:
    try:
        return datetime.fromisoformat(re.sub(r"(\.[0-9]{6})[0-9]+", r"\1", str(text)).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _teardown(action: _Action, descriptor: Mapping[str, Any], *, outcome: str, terminal: str, wait: bool,
              usage: tuple[float, int] | None = None) -> dict[str, Any]:
    """Prove compute-zero, then provider-zero once the write URLs have expired; seal the teardown, turn the
    lease terminal (which frees its slot) and settle the attempt's spend."""
    runtime, root, job_id, attempt_id = action.runtime, action.root, descriptor["job_id"], descriptor["attempt_id"]
    while True:
        lease, now = _lease(root, job_id), float(runtime.clock())
        compute = prove_compute_zero(action, lease, descriptor)
        if not compute_zero_proven(compute, worker_identity=lease["worker_identity"]):
            return {"status": "teardown_pending", "blockers": ["remote_cpu_compute_zero_unproven"]}
        record = teardown_record(descriptor=descriptor, worker_identity=lease["worker_identity"], outcome=outcome,
                                 compute=compute, provider=prove_provider_zero(action, lease, descriptor),
                                 observed_at_epoch=now)
        if record["provider_zero_proven"] or not wait:
            break
        runtime.sleep(max(1.0, lease["write_urls_expire_at_epoch"] - now))
    if not record["provider_zero_proven"]:
        leases.transition(root, job_id, attempt_id=attempt_id, to_state=None, now=now, updates={"compute_zero": compute})
        return {"status": "teardown_pending", "blockers": ["remote_cpu_provider_zero_unproven"]}
    write_remote_cpu_record(root / "teardowns" / f"{attempt_id}.json", record)
    leases.transition(root, job_id, attempt_id=attempt_id, to_state=terminal, now=now,
                      updates={"compute_zero": compute, "teardown": record, "outcome": outcome})
    if lease["worker_identity"] is None and compute["executions_for_attempt"] == 0:
        settled, basis = 0.0, "no_execution"
    elif usage is not None:
        limits = {**descriptor["limits"], "task_timeout_seconds": usage[0], "max_output_bytes": usage[1]}
        settled, basis = worst_case_usd(limits=limits, rate_table=action.config["rate_table"]), "execution_runtime"
    else:
        settled, basis = worst_case_usd(limits=descriptor["limits"], rate_table=action.config["rate_table"]), "worst_case"
    settle_remote_cpu_attempt(descriptor=descriptor, teardown_digest=record["teardown_digest"], settled_usd=settled,
                              basis=basis, now=now)
    return {"status": terminal, "blockers": [], "settled_usd": settled,
            "teardown": {name: record[name] for name in ("teardown_digest", "compute_zero_proven",
                                                          "provider_zero_proven", "observed_at_epoch")}}


def _probe_descriptor(action: _Action, stage: str, *, source: Mapping[str, Any], host_digest: str) -> dict[str, Any]:
    """Stage the probe request as the queue envelope and seal an ``environment_probe`` descriptor."""
    entry, (client, bucket, _) = action.config["stages"][PROBE_STAGE], action.runtime.object_store
    request = record_bytes({"schema_version": PROBE_REQUEST_SCHEMA_VERSION, "stage": stage, "job": entry["job"],
                            "image": entry["image"], "requested_at_epoch": action.now, "nonce": secrets.token_hex(16)})
    digest = hashlib.sha256(request).hexdigest()
    name = f"probe-{STAGES[stage]['abbreviation']}-{int(action.now)}-{digest}.json"
    staged = publish_configured_scene_stream(
        write_stream=lambda sink: sink.write(request), digest=f"sha256:{digest}", size_bytes=len(request),
        filename=name, artifact_kind="remote-cpu-input", client=client, bucket=bucket)
    limits = stage_limits(action.config, PROBE_STAGE)
    return build_descriptor(
        config=action.config, stage=PROBE_STAGE, mode="shadow", attempt=1,
        queue_row={"queue": STAGES[PROBE_STAGE]["queue"], "name": name, "envelope_digest": f"sha256:{digest}"},
        code={"source_commit": source["source_commit"], "image": entry["image"], "environment_digest": host_digest,
              "source_archive": {field: source[field] for field in ("digest", "size_bytes", "uri")}},
        environment={}, inputs=[{"role": "queue_envelope", "contract_path": PROBE_REQUEST_SCHEMA_VERSION,
                                 "digest": f"sha256:{digest}", "size_bytes": len(request), "mode": "0440",
                                 "materialize_at": f"{PROBE_ROOT}/requests/{name}", "uri": staged["uri"]}],
        outputs={"output_root": f"{PROBE_ROOT}/outputs/{name.removesuffix(f'-{digest}.json')}",
                 "declared_scratch": [], "object_prefix": f"s3://{bucket}/{DEFAULT_KEY_PREFIX}"},
        limits=limits, closure={"class": "not_applicable", "source_appearance_digest": None},
        spend={"worst_case_usd": worst_case_usd(limits=limits, rate_table=action.config["rate_table"]),
               "rate_table_digest": canonical_digest(action.config["rate_table"])})


def _await_execution(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Poll the probe to a terminal state, renewing its lease on each advancing heartbeat and cancelling it
    (termination only) once its hard deadline passes."""
    runtime, store, lease = action.runtime, action.runtime.object_store, _lease(action.root, descriptor["job_id"])
    name = execution_resource_name(run_target(descriptor)["job"], execution_name_of(lease["worker_identity"]))
    hard, cancelled = lease["deadlines"]["hard_deadline_epoch"], False
    while True:
        execution, now = runtime.cloud_run.get_execution(name), float(runtime.clock())
        if execution.get("completionTime"):
            return execution
        try:
            raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "heartbeat.json",
                                                 maximum_size_bytes=MAX_HEARTBEAT_BYTES, client=store[0], bucket=store[1])
            leases.observe_heartbeat(action.root, descriptor["job_id"], json.loads(raw or b"null"),
                                     execution_running=bool(execution.get("runningCount")), now=now)
        except (ValueError, TaskEvaluationConfiguredSceneObjectStoreError):  # none of these renews the lease
            pass
        if now >= hard and not cancelled:
            runtime.cloud_run.cancel_execution(name)
            cancelled = True
        elif now >= hard + CANCEL_GRACE_SECONDS:
            raise RemoteCpuAllocatorError("remote_cpu_execution_not_terminal")
        runtime.sleep(runtime.poll_seconds)


def _collect_probe(action: _Action, descriptor: Mapping[str, Any], execution: Mapping[str, Any], *, stage: str,
                   host: Mapping[str, Any]) -> dict[str, Any]:
    """Fence the probe's receipt, record the worker environment, and move the lease to collecting."""
    store, lease = action.runtime.object_store, _lease(action.root, descriptor["job_id"])
    raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "receipt.json",
                                         maximum_size_bytes=MAX_RECEIPT_BYTES, client=store[0], bucket=store[1])
    outcome, receipt, environment = "probe_receipt_missing", None, None
    if raw is not None:
        try:
            receipt = validate_receipt(json.loads(raw), descriptor=descriptor,
                                       execution_name=execution_name_of(lease["worker_identity"]))["receipt"]
            worker, outcome = (receipt["result"] or {}).get("environment"), "probe_result_invalid"
            if isinstance(worker, Mapping) and worker.get("environment_digest") == receipt["environment"][
                    "environment_digest"]:
                environment, outcome = _record_environment(action, descriptor, stage=stage, worker=worker, host=host,
                                                           receipt=receipt), "environment_recorded"
        except ValueError:  # a malformed or fenced receipt: RemoteCpuContractError is a ValueError
            outcome = "probe_receipt_invalid"
    if lease["state"] in ("dispatched", "running"):
        leases.transition(action.root, descriptor["job_id"], attempt_id=descriptor["attempt_id"], to_state="collecting",
                          now=float(action.runtime.clock()), updates={"outcome": outcome})
    created, completed = _epoch(execution.get("createTime")), _epoch(execution.get("completionTime"))
    timeout = float(descriptor["limits"]["task_timeout_seconds"])
    seconds = min(timeout, max(0.0, completed - created)) if created and completed else timeout
    uploaded = receipt["bytes_uploaded"] if receipt else descriptor["limits"]["max_output_bytes"]
    return {"outcome": outcome, "environment": environment, "usage": (seconds, uploaded),
            "terminal": "fallback_host" if lease["state"] == "expired" else "completed" if environment else "blocked"}


def _record_environment(action: _Action, descriptor: Mapping[str, Any], *, stage: str, worker: Mapping[str, Any],
                        host: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Record the worker environment for this image, with its parity against the host field by field."""
    record = {"schema_version": ENVIRONMENT_SCHEMA_VERSION, "stage": stage, "job": descriptor["execution"]["job"],
              "image": descriptor["code"]["image"], "environment_digest": worker["environment_digest"],
              "cpu_class": receipt["environment"]["cpu_class"], "host_environment_digest": host.get("environment_digest"),
              "parity": {name: worker.get(name) == host.get(name) for name in DIGESTED_FIELDS},
              "worker_environment": dict(worker), "probe_attempt_id": descriptor["attempt_id"],
              "receipt_digest": receipt["receipt_digest"], "recorded_at_epoch": float(action.runtime.clock())}
    _write_result(action.root / "environment" / f"{stage}.json", record)
    return {name: record[name] for name in ("environment_digest", "cpu_class", "host_environment_digest", "parity")}


def preflight_remote_cpu_stage(action: _Action, stage: str) -> dict[str, Any]:
    """The owner's preflight (plan 14 §8): a real ``environment_probe`` attempt on ``stage``'s job - admitted,
    consumed, leased, sentinel- and ``validate_only``-checked, run once, collected and torn down."""
    runtime = action.runtime
    if runtime.stage_release_source is None:
        return {"status": "blocked", "blockers": ["remote_cpu_release_source_staging_unavailable"]}
    action.authority, action.blockers = load_standing_authority(stage=stage, now=action.now)
    if action.blockers or not action.execute:
        return {"status": "blocked" if action.blockers else "dry_run_ready", "blockers": action.blockers}
    host = dict(runtime.host_environment())
    probe = replace(action, config=_probe_config(action.config, stage))
    descriptor = _probe_descriptor(probe, stage, source=runtime.stage_release_source(),
                                   host_digest=host["environment_digest"])
    write_remote_cpu_record(action.root / "descriptors" / f"{descriptor['attempt_id']}.json", descriptor)
    result = {"probe": {name: descriptor[name] for name in ("job_id", "attempt_id", "descriptor_digest")},
              **_dispatch(probe, descriptor)}
    if result["status"] != "dispatched":
        return result
    collected = _collect_probe(probe, descriptor, _await_execution(probe, descriptor), stage=stage, host=host)
    torn = _teardown(probe, descriptor, outcome=collected["outcome"], terminal=collected["terminal"], wait=True,
                     usage=collected["usage"])
    blockers = sorted({*torn["blockers"], *([] if collected["environment"] else [f"remote_cpu_{collected['outcome']}"])})
    return {**result, **torn, "outcome": collected["outcome"], "environment": collected["environment"],
            "status": "completed" if torn["status"] == "completed" and not blockers else "blocked", "blockers": blockers}


def _terminate(action: _Action, executions: list[Mapping[str, Any]], status: str) -> dict[str, Any]:
    """Cancel each execution; one that finished in the meantime is left as it is."""
    names = [str(row.get("name") or "") for row in executions]
    if not action.execute:
        return {"status": "dry_run_ready", "would_cancel": [name.rsplit("/", 1)[-1] for name in names], "blockers": []}
    cancelled, finished, blockers = [], [], []
    for name in names:
        try:
            action.runtime.cloud_run.cancel_execution(name)
            cancelled.append(name.rsplit("/", 1)[-1])
        except CloudRunJobsError as exc:
            if exc.code == "FAILED_PRECONDITION":
                finished.append(name.rsplit("/", 1)[-1])
            else:
                blockers.append(f"remote_cpu_cancel_failed:{exc.code}")
    return {"status": "blocked" if blockers else status, "cancelled": cancelled, "already_finished": finished,
            "blockers": sorted(set(blockers))}


def cancel_remote_cpu_attempt(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Terminate every unfinished execution carrying this attempt; this action never starts anything."""
    rows, pages = action.runtime.cloud_run.list_all_executions(run_target(descriptor)["job"])
    running = [row for row in rows
               if env_value(row, ATTEMPT_VARIABLE) == descriptor["attempt_id"] and not row.get("completionTime")]
    return {**_terminate(action, running, "cancelled"), "listing_pages": pages}


def _owned(row: Mapping[str, Any], owners: Mapping[str, Mapping[str, Any]]) -> bool:
    lease = owners.get(env_value(row, ATTEMPT_VARIABLE) or "")
    if lease is None:
        return False
    # While dispatching, the lease has no worker identity yet and its run may be this one.
    return lease["worker_identity"] is None or lease["worker_identity"].rsplit("/", 1)[-1] == str(
        row.get("name") or "").rsplit("/", 1)[-1]


def sweep_remote_cpu_stage(action: _Action, stage: str) -> dict[str, Any]:
    """Cancel every unfinished execution of the stage's job that no live lease owns (plan 14 §11)."""
    config = action.config
    job = job_resource_name(project=config["project"], region=config["region"], job=config["stages"][stage]["job"])
    rows, pages = action.runtime.cloud_run.list_all_executions(job)
    owners = {lease["attempt_id"]: lease for lease in leases.live_leases(action.root, now=action.now)}
    orphans = [row for row in rows if not row.get("completionTime") and not _owned(row, owners)]
    return {**_terminate(action, orphans, "swept"), "executions_listed": len(rows), "listing_pages": pages,
            "live_leases": len(owners)}


def _reconcile(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a dispatch whose response was lost, and finish teardowns left waiting (plan 14 §10, §11)."""
    lease = _lease(action.root, descriptor["job_id"])
    if lease is None or lease["attempt_id"] != descriptor["attempt_id"]:
        return {"status": "nothing_to_reconcile", "blockers": []}
    state, identity, deadlines = lease["state"], lease["worker_identity"], lease["deadlines"] or {}
    lost = identity is None and lease["dispatch_started"] and state in ("dispatching", "expired")
    # A probe is its preflight's to close; reconcile takes over only once that preflight would have finished.
    probe = (descriptor["stage"] == PROBE_STAGE and identity is not None and state in ("dispatched", "running",
             "expired", "collecting") and action.now >= lease["write_urls_expire_at_epoch"] + RECONCILE_AFTER_SECONDS)
    # A dispatch in flight in another process is never second-guessed.
    in_flight = state == "dispatching" and action.now < deadlines.get("dispatch_started_at_epoch", 0) + RECONCILE_AFTER_SECONDS
    if not (lost or probe) or in_flight:
        return {"status": "nothing_to_reconcile", "blockers": [], "lease_state": state}
    if not action.execute:
        return {"status": "dry_run_ready", "blockers": [], "lease_state": state}
    if probe:
        name = execution_resource_name(run_target(descriptor)["job"], execution_name_of(identity))
        if state != "collecting" and not action.runtime.cloud_run.get_execution(name).get("completionTime"):
            action.runtime.cloud_run.cancel_execution(name)
            return {"status": "teardown_pending", "blockers": ["remote_cpu_compute_zero_unproven"], "lease_state": state}
        outcome = "probe_interrupted" if state in ("dispatched", "running") else lease["outcome"]
        if state in ("dispatched", "running"):
            leases.transition(action.root, descriptor["job_id"], attempt_id=descriptor["attempt_id"],
                              to_state="collecting", now=action.now, updates={"outcome": outcome})
        return _teardown(action, descriptor, outcome=outcome, wait=False, terminal="fallback_host" if state == "expired"
                         else "completed" if outcome == "environment_recorded" else "blocked")
    found = reconcile_ambiguous_dispatch(action, descriptor)
    if found["status"] == "unresolved":
        return {"status": "ambiguous_dispatch_unresolved", "blockers": found["blockers"], "reconciled": found}
    if found["executions"] and state == "dispatching":
        return {**_record_dispatched(action, descriptor, found["executions"][0]), "reconciled": found}
    if found["executions"]:  # it started after its lease expired: bind it, then it closes as a fallback
        leases.transition(action.root, descriptor["job_id"], attempt_id=descriptor["attempt_id"], to_state=None,
                          now=action.now, updates={"worker_identity": worker_identity_for(
                              descriptor["execution"], found["executions"][0].rsplit("/", 1)[-1])})
    return {**_teardown(action, descriptor, outcome=lease["outcome"] or "remote_cpu_dispatch_lost", wait=False,
                        terminal="fallback_host" if found["executions"] else "abandoned_dispatch"), "reconciled": found}


def add_remote_cpu_job_arguments(commands: Any) -> None:
    """Register ``remote-cpu-job`` on the canonical paid allocator (plan 14 §8)."""
    command = commands.add_parser("remote-cpu-job", help="Admit, run, reconcile or terminate remote CPU jobs.")
    command.add_argument("--action", choices=ACTIONS, required=True)
    command.add_argument("--stage", choices=sorted(set(STAGES) - {PROBE_STAGE}), required=True)
    command.add_argument("--descriptor", help="A sealed remote_cpu_job_descriptor.v1 (dispatch, reconcile, cancel).")
    command.add_argument("--lease", required=True, help="The remote CPU job store root that holds the leases.")
    command.add_argument("--out", required=True, help="Where this action's door-safe result is written.")
    command.add_argument("--execute", action="store_true")


def _run_action(args: argparse.Namespace, runtime: RemoteCpuRuntime, now: float) -> dict[str, Any]:
    config, blockers = load_remote_cpu_config(runtime.config_path)
    if not blockers and args.stage not in config["stages"]:
        blockers = [f"remote_cpu_config_invalid:stage_missing:{args.stage}"]
    blockers = blockers or _connect(runtime, config)
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    action = _Action(runtime=runtime, config=config, root=Path(args.lease), execute=bool(args.execute), now=now)
    if args.action == "sweep":
        return sweep_remote_cpu_stage(action, args.stage)
    if args.action == "preflight":
        return preflight_remote_cpu_stage(action, args.stage)
    if not args.descriptor:
        return {"status": "blocked", "blockers": ["remote_cpu_descriptor_missing"]}
    try:
        raw = _read_private_json(args.descriptor, forbidden_mode=0o022)
    except RemoteCpuAllocatorError as exc:
        return {"status": "blocked", "blockers": [f"remote_cpu_descriptor_invalid:{exc.code}"]}
    action.config, stage = _config_for(config, raw)
    descriptor = validate_descriptor(raw, config=action.config)
    if stage != args.stage or (args.action == "dispatch" and descriptor["stage"] == PROBE_STAGE):
        return {"status": "blocked", "blockers": ["remote_cpu_descriptor_stage_mismatch"]}
    if args.action == "cancel":
        return cancel_remote_cpu_attempt(action, descriptor)
    if args.action == "reconcile":
        return _reconcile(action, descriptor)
    action.authority, action.blockers = load_standing_authority(stage=stage, now=now)
    return _dispatch(action, descriptor)


_SUCCESS = {"dispatch": {"dispatched", "dry_run_ready"}, "cancel": {"cancelled", "dry_run_ready"},
            "sweep": {"swept", "dry_run_ready"}, "preflight": {"completed", "dry_run_ready"},
            "reconcile": {"dispatched", "abandoned_dispatch", "fallback_host", "completed", "teardown_pending",
                          "nothing_to_reconcile", "dry_run_ready"}}


def run_remote_cpu_job(args: argparse.Namespace, *, runtime: RemoteCpuRuntime | None = None) -> dict[str, Any]:
    """Run one ``remote-cpu-job`` action; the result is written to ``--out`` and carries ``success``."""
    runtime = runtime or RemoteCpuRuntime()
    now = float(runtime.clock())
    result: dict[str, Any] = {"schema_version": RESULT_SCHEMA_VERSION, "action": args.action, "stage": args.stage,
                              "execute": bool(args.execute), "observed_at_epoch": now, "status": "blocked",
                              "blockers": []}
    try:
        result.update(_run_action(args, runtime, now))
    except RemoteCpuContractError as exc:
        result.update(status="blocked", blockers=list(exc.reasons))
    except (CloudRunJobsError, RemoteCpuAllocatorError) as exc:
        result.update(status="blocked", blockers=[exc.code])
    except PaidResourceAdmissionBlocked as exc:
        result.update(status="blocked", blockers=list(exc.blockers))
    except Exception as exc:  # noqa: BLE001 - fail closed with a typed record; the cause may name a URL
        result.update(status="blocked", blockers=[f"remote_cpu_{args.action}_failed:{type(exc).__name__}"])
    result["success"] = result["status"] in _SUCCESS.get(args.action, ())
    try:
        _write_result(Path(args.out), result)
    except RemoteCpuContractError:
        result = {"schema_version": RESULT_SCHEMA_VERSION, "action": args.action, "stage": args.stage,
                  "status": "blocked", "blockers": ["remote_cpu_result_unrecordable"], "success": False}
        _write_result(Path(args.out), result)
    return result
