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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import remote_cpu_job_lease as leases
from .cloud_run_jobs_client import (
    CLOUD_RUN_CPU_JOB_RESOURCE_CLASS,
    CloudRunJobsClient,
    CloudRunJobsError,
    GcsTransportBucket,
    allocation_binding,
    job_definition_blockers,
    load_dispatcher_credentials,
    run_target,
)
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .remote_cpu_job_contract import (
    CONFIG_SCHEMA_VERSION,
    MAX_ATTEMPTS_CAP,
    STAGES,
    RemoteCpuContractError,
    record_bytes,
    validate_descriptor,
)
from .remote_cpu_job_records import fsync_directory
from .spend_authority_consumption_root import (
    SpendAuthorityRootError,
    authorizations_root,
    prepare_consumption_root,
    spend_authority_root,
)
from .task_evaluation_configured_scene_object_store import (
    TaskEvaluationConfiguredSceneObjectStoreError,
    remote_cpu_object_store,
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
SETTLED_DIRECTORY = "remote-cpu-settled"
LEDGER_LOCK = "remote-cpu.lock"
DAY_SECONDS = 86400
GIB = 1024**3
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


def _count(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _amount(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 1024


def _micros(value: float) -> int:
    return round(float(value) * 1_000_000)


def _iso_date(value: Any) -> bool:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


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


def _create_once(path: Path, value: Mapping[str, Any]) -> bool:
    """O_EXCL plus a link, fsynced and 0600: ``False`` when ``path`` already exists."""

    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)
    return True


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Atomically replace ``--out``; the record guard refuses any URL or credential-shaped content."""

    payload = record_bytes(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), 0o640)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_entry_valid(stage: str, entry: Any) -> bool:
    return (stage in STAGES and stage != PROBE_STAGE and isinstance(entry, Mapping) and set(entry) == _STAGE_KEYS
            and isinstance(entry["job"], str) and _JOB.fullmatch(entry["job"]) is not None
            and isinstance(entry["image"], str) and _IMAGE.fullmatch(entry["image"]) is not None
            and all(_count(entry[name], 1) for name in ("vcpu", "memory_bytes", "ephemeral_bytes",
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
        "transport_bucket": not isinstance(config.get("transport_bucket"), str)
        or _BUCKET.fullmatch(config["transport_bucket"]) is None,
        "stages": not stages or not all(_stage_entry_valid(stage, entry) for stage, entry in stages.items()),
        "rate_table": set(rates) != {"source", "observed_on", *_RATES} or not all(_amount(rates[n]) for n in _RATES)
        or not _text(rates["source"]) or not _iso_date(rates["observed_on"]),
        "max_live_executions": not _count(config.get("max_live_executions"), 1),
        "max_attempts": not _count(config.get("max_attempts"), 1) or config["max_attempts"] > MAX_ATTEMPTS_CAP,
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
        "max_executions": not _count(authority.get("max_executions"), 1),
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


def _admission_view(admission: Mapping[str, Any]) -> dict[str, Any]:
    return {name: admission[name] for name in ("schema_version", "status", "resource_class", "blockers",
                                               "allocation_binding", "allocation_binding_digest")}


@dataclass
class RemoteCpuRuntime:
    """What the seam touches beyond this host; production builds each service lazily from its credential."""

    config_path: str | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    cloud_run: Any = None
    transport_bucket: Any = None
    object_store: tuple[Any, str, str] | None = None


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


def _dispatch(action: _Action, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Admit, lease and consume one attempt; its transport and execution follow only after admission."""

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
        result = {"admission": _admission_view(admission), "blockers": admission["blockers"], "job_id": job_id,
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
    return {**result, "status": "admitted"}


def _run_action(args: argparse.Namespace, runtime: RemoteCpuRuntime, now: float) -> dict[str, Any]:
    config, blockers = load_remote_cpu_config(runtime.config_path)
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    if args.stage not in config["stages"]:
        return {"status": "blocked", "blockers": [f"remote_cpu_config_invalid:stage_missing:{args.stage}"]}
    blockers = _connect(runtime, config)
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    action = _Action(runtime=runtime, config=config, root=Path(args.lease), execute=bool(args.execute), now=now)
    if not args.descriptor:
        return {"status": "blocked", "blockers": ["remote_cpu_descriptor_missing"]}
    try:
        descriptor = validate_descriptor(_read_private_json(args.descriptor, forbidden_mode=0o022), config=config)
    except RemoteCpuAllocatorError as exc:
        return {"status": "blocked", "blockers": [f"remote_cpu_descriptor_invalid:{exc.code}"]}
    if descriptor["stage"] != args.stage:
        return {"status": "blocked", "blockers": ["remote_cpu_descriptor_stage_mismatch"]}
    action.authority, action.blockers = load_standing_authority(stage=descriptor["stage"], now=now)
    return _dispatch(action, descriptor)


_SUCCESS = {"dispatch": {"dispatched", "dry_run_ready"}}


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
        result = {**{name: result[name] for name in ("schema_version", "action", "stage", "execute",
                                                      "observed_at_epoch")},
                  "status": "blocked", "blockers": ["remote_cpu_result_unrecordable"], "success": False}
        _write_result(Path(args.out), result)
    return result
