"""Production warm-GPU worker registry and customer binding control plane.

Customer requests may lease an already-ready worker or enqueue asynchronous
scale demand.  They never allocate a VM, install host software, pull a worker
image, or warm Isaac/model state inline.  SQLite provides restart-safe atomic
leases for a single control-plane deployment; the API stays provider-neutral.

The registry proves control-plane behavior only.  Customer launch readiness
also requires fresh provider evidence for the exact host image and worker
digest, measured warm-bind and cold-replenishment SLOs, rollback, and teardown.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .common import read_json_any, write_json

PRODUCTION_GPU_WORKER_POOL_SCHEMA_VERSION = "production_gpu_worker_pool.v1"
PRODUCTION_GPU_WORKER_SCHEMA_VERSION = "production_gpu_worker.v1"
PRODUCTION_GPU_BINDING_SCHEMA_VERSION = "production_gpu_worker_binding.v1"
PRODUCTION_GPU_SCALE_REQUEST_SCHEMA_VERSION = "production_gpu_scale_request.v1"
PRODUCTION_GPU_STARTUP_READINESS_SCHEMA_VERSION = "production_gpu_startup_readiness.v1"
POOL_TOKEN_FILE_ENV = "BLUEPRINT_PRODUCTION_GPU_POOL_TOKEN_FILE"

_WORKER_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_JOB_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_SHA256_IMAGE = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")
_LEASE_TOKEN = re.compile(r"\Agwl_[0-9a-f]{64}\Z")
_SCALE_TOKEN = re.compile(r"\Agsc_[0-9a-f]{64}\Z")
REQUIRED_READY_CHECKS = (
    "host_image_booted",
    "nvidia_driver_ready",
    "container_runtime_ready",
    "worker_image_cached",
    "models_cached_offline",
    "isaac_renderer_warm",
    "kitchen_scene_loaded",
    "policy_endpoint_ready",
    "worker_healthcheck_passed",
)


class WorkerPoolError(RuntimeError):
    """Base error for production pool contract violations."""


class WorkerLeaseConflict(WorkerPoolError):
    """A worker lease transition did not carry the active lease token."""


class UnknownWorker(WorkerPoolError):
    """The requested worker is not registered."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded(value: Any, *, field: str, maximum: int) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"{field}_required")
    if len(text.encode("utf-8")) > maximum:
        raise ValueError(f"{field}_too_large")
    return text


def _worker_id(value: Any) -> str:
    text = _bounded(value, field="worker_id", maximum=128)
    if not _WORKER_ID.fullmatch(text):
        raise ValueError("worker_id_invalid")
    return text


def _job_id(value: Any) -> str:
    text = _bounded(value, field="job_id", maximum=192)
    if not _JOB_ID.fullmatch(text):
        raise ValueError("job_id_invalid")
    return text


def _image_ref(value: Any) -> str:
    text = _bounded(value, field="worker_image_ref", maximum=512)
    if not _SHA256_IMAGE.fullmatch(text):
        raise ValueError("worker_image_ref_must_be_digest_pinned")
    return text


def _lease_token(value: Any) -> str:
    text = _bounded(value, field="lease_token", maximum=68)
    if not _LEASE_TOKEN.fullmatch(text):
        raise ValueError("lease_token_invalid")
    return text


def release_fingerprint(*, host_image_id: str, worker_image_ref: str, gpu_family: str) -> str:
    host = _bounded(host_image_id, field="host_image_id", maximum=512)
    image = _image_ref(worker_image_ref)
    gpu = _bounded(gpu_family, field="gpu_family", maximum=128)
    return _digest(_canonical_json({"gpu_family": gpu, "host_image_id": host, "worker_image_ref": image}))


class ProductionGpuWorkerPool:
    """Restart-safe exact-release worker registry with atomic customer leases."""

    def __init__(self, database_path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS production_gpu_workers (
                    worker_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    host_image_id TEXT NOT NULL,
                    worker_image_ref TEXT NOT NULL,
                    gpu_family TEXT NOT NULL,
                    release_fingerprint TEXT NOT NULL,
                    endpoint_ref TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('ready','leased','draining','quarantined')),
                    readiness_json TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    lease_job_id TEXT,
                    lease_token_digest TEXT,
                    lease_expires_at REAL,
                    transition_reason TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS production_gpu_workers_selection
                    ON production_gpu_workers(release_fingerprint, state, heartbeat_at, registered_at);
                CREATE UNIQUE INDEX IF NOT EXISTS production_gpu_workers_job_lease
                    ON production_gpu_workers(lease_job_id) WHERE lease_job_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS production_gpu_scale_requests (
                    scale_request_id TEXT PRIMARY KEY,
                    release_fingerprint TEXT NOT NULL,
                    host_image_id TEXT NOT NULL,
                    worker_image_ref TEXT NOT NULL,
                    gpu_family TEXT NOT NULL,
                    requested_ready_workers INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending','claimed','satisfied','cancelled')),
                    claim_owner TEXT,
                    claim_token_digest TEXT,
                    claim_expires_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS production_gpu_scale_pending_release
                    ON production_gpu_scale_requests(release_fingerprint)
                    WHERE status IN ('pending','claimed');
                """
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_readiness(readiness: Mapping[str, Any]) -> dict[str, bool]:
        if not isinstance(readiness, Mapping):
            raise ValueError("readiness_must_be_mapping")
        normalized = {name: readiness.get(name) is True for name in REQUIRED_READY_CHECKS}
        missing = [name for name, passed in normalized.items() if not passed]
        if missing:
            raise ValueError("worker_readiness_incomplete:" + ",".join(missing))
        return normalized

    def register_ready_worker(
        self,
        *,
        worker_id: str,
        provider: str,
        host_image_id: str,
        worker_image_ref: str,
        gpu_family: str,
        endpoint_ref: str,
        readiness: Mapping[str, Any],
    ) -> dict[str, Any]:
        worker = _worker_id(worker_id)
        provider_name = _bounded(provider, field="provider", maximum=64)
        host = _bounded(host_image_id, field="host_image_id", maximum=512)
        image = _image_ref(worker_image_ref)
        gpu = _bounded(gpu_family, field="gpu_family", maximum=128)
        endpoint = _bounded(endpoint_ref, field="endpoint_ref", maximum=512)
        ready = self._validate_readiness(readiness)
        fingerprint = release_fingerprint(host_image_id=host, worker_image_ref=image, gpu_family=gpu)
        now = float(self._clock())
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT state, release_fingerprint FROM production_gpu_workers WHERE worker_id = ?",
                (worker,),
            ).fetchone()
            if existing is not None and existing["state"] == "leased":
                raise WorkerLeaseConflict("leased_worker_cannot_reregister")
            if existing is not None and existing["release_fingerprint"] != fingerprint:
                raise WorkerLeaseConflict("worker_identity_release_changed")
            connection.execute(
                """
                INSERT INTO production_gpu_workers (
                    worker_id, provider, host_image_id, worker_image_ref, gpu_family,
                    release_fingerprint, endpoint_ref, state, readiness_json,
                    registered_at, heartbeat_at, transition_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, 'readiness_passed')
                ON CONFLICT(worker_id) DO UPDATE SET
                    provider=excluded.provider, endpoint_ref=excluded.endpoint_ref,
                    state='ready', readiness_json=excluded.readiness_json,
                    heartbeat_at=excluded.heartbeat_at, lease_job_id=NULL,
                    lease_token_digest=NULL, lease_expires_at=NULL,
                    transition_reason='readiness_reconfirmed'
                """,
                (worker, provider_name, host, image, gpu, fingerprint, endpoint, _canonical_json(ready), now, now),
            )
            connection.execute(
                """
                UPDATE production_gpu_scale_requests
                   SET requested_ready_workers = CASE
                           WHEN requested_ready_workers > 1
                           THEN requested_ready_workers - 1 ELSE 0 END,
                       status = CASE
                           WHEN requested_ready_workers > 1
                           THEN status ELSE 'satisfied' END,
                       updated_at = ?,
                       claim_owner = CASE
                           WHEN requested_ready_workers > 1
                           THEN claim_owner ELSE NULL END,
                       claim_token_digest = CASE
                           WHEN requested_ready_workers > 1
                           THEN claim_token_digest ELSE NULL END,
                       claim_expires_at = CASE
                           WHEN requested_ready_workers > 1
                           THEN claim_expires_at ELSE NULL END
                 WHERE release_fingerprint=? AND status IN ('pending','claimed')
                """,
                (now, fingerprint),
            )
        return {
            "schema_version": PRODUCTION_GPU_WORKER_SCHEMA_VERSION,
            "worker_id": worker,
            "state": "ready",
            "release_fingerprint": fingerprint,
            "ready_for_customer_binding": True,
        }

    def heartbeat(
        self,
        *,
        worker_id: str,
        lease_token: str | None = None,
        trusted_worker_agent: bool = False,
    ) -> dict[str, Any]:
        worker = _worker_id(worker_id)
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state, lease_token_digest FROM production_gpu_workers WHERE worker_id=?",
                (worker,),
            ).fetchone()
            if row is None:
                raise UnknownWorker("worker_unknown")
            if row["state"] == "leased" and not trusted_worker_agent:
                token = _lease_token(lease_token)
                if not hmac.compare_digest(str(row["lease_token_digest"] or ""), _digest(token)):
                    raise WorkerLeaseConflict("active_lease_token_required")
            connection.execute(
                "UPDATE production_gpu_workers SET heartbeat_at=? WHERE worker_id=?",
                (now, worker),
            )
        return {"worker_id": worker, "state": row["state"], "heartbeat_recorded": True}

    def quarantine_worker(self, *, worker_id: str, reason: str) -> dict[str, Any]:
        """Remove a provider worker from customer selection before teardown."""

        worker = _worker_id(worker_id)
        transition = _bounded(reason, field="quarantine_reason", maximum=160)
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM production_gpu_workers WHERE worker_id=?", (worker,)
            ).fetchone()
            if row is None:
                raise UnknownWorker("worker_unknown")
            connection.execute(
                "UPDATE production_gpu_workers SET state='quarantined', heartbeat_at=?, "
                "lease_job_id=NULL, lease_token_digest=NULL, lease_expires_at=NULL, "
                "transition_reason=? WHERE worker_id=?",
                (now, transition, worker),
            )
        return {
            "worker_id": worker,
            "previous_state": row["state"],
            "state": "quarantined",
            "ready_for_customer_binding": False,
        }

    @staticmethod
    def _expire_stale(connection: sqlite3.Connection, *, now: float, heartbeat_ttl_seconds: float) -> None:
        connection.execute(
            "UPDATE production_gpu_workers SET state='quarantined', transition_reason='heartbeat_stale' "
            "WHERE state='ready' AND heartbeat_at < ?",
            (now - heartbeat_ttl_seconds,),
        )
        connection.execute(
            "UPDATE production_gpu_workers SET state='quarantined', transition_reason='lease_expired', "
            "lease_job_id=NULL, lease_token_digest=NULL, lease_expires_at=NULL "
            "WHERE state='leased' AND lease_expires_at <= ?",
            (now,),
        )

    @staticmethod
    def _scale_request(
        connection: sqlite3.Connection,
        *,
        fingerprint: str,
        host_image_id: str,
        worker_image_ref: str,
        gpu_family: str,
        requested_ready_workers: int,
        reason: str,
        now: float,
    ) -> str:
        existing = connection.execute(
            "SELECT scale_request_id, requested_ready_workers FROM production_gpu_scale_requests "
            "WHERE release_fingerprint=? AND status IN ('pending','claimed')",
            (fingerprint,),
        ).fetchone()
        if existing is not None:
            connection.execute(
                "UPDATE production_gpu_scale_requests SET requested_ready_workers=?, reason=?, updated_at=? "
                "WHERE scale_request_id=?",
                (max(int(existing["requested_ready_workers"]), requested_ready_workers), reason, now, existing["scale_request_id"]),
            )
            return str(existing["scale_request_id"])
        request_id = f"gps_{uuid.uuid4().hex}"
        connection.execute(
            """
            INSERT INTO production_gpu_scale_requests (
                scale_request_id, release_fingerprint, host_image_id,
                worker_image_ref, gpu_family, requested_ready_workers,
                reason, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (request_id, fingerprint, host_image_id, worker_image_ref, gpu_family, requested_ready_workers, reason, now, now),
        )
        return request_id

    def claim_scale_request(
        self,
        *,
        autoscaler_id: str,
        lease_seconds: float = 120.0,
    ) -> dict[str, Any] | None:
        """Atomically lease one capacity deficit to an asynchronous reconciler."""

        owner = _bounded(autoscaler_id, field="autoscaler_id", maximum=128)
        if not _WORKER_ID.fullmatch(owner):
            raise ValueError("autoscaler_id_invalid")
        lease = float(lease_seconds)
        if not 10.0 <= lease <= 900.0:
            raise ValueError("scale_lease_seconds_out_of_range")
        now = float(self._clock())
        token = f"gsc_{secrets.token_hex(32)}"
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM production_gpu_scale_requests
                 WHERE status='pending'
                    OR (status='claimed' AND claim_expires_at <= ?)
                 ORDER BY created_at, scale_request_id
                 LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """
                UPDATE production_gpu_scale_requests
                   SET status='claimed', claim_owner=?, claim_token_digest=?,
                       claim_expires_at=?, updated_at=?
                 WHERE scale_request_id=?
                   AND (status='pending' OR (status='claimed' AND claim_expires_at <= ?))
                """,
                (owner, _digest(token), now + lease, now, row["scale_request_id"], now),
            )
            if updated.rowcount != 1:
                raise WorkerLeaseConflict("atomic_scale_request_claim_lost")
        return {
            "schema_version": PRODUCTION_GPU_SCALE_REQUEST_SCHEMA_VERSION,
            "scale_request_id": row["scale_request_id"],
            "status": "claimed",
            "autoscaler_id": owner,
            "scale_token": token,
            "claim_expires_at_epoch": now + lease,
            "host_image_id": row["host_image_id"],
            "worker_image_ref": row["worker_image_ref"],
            "gpu_family": row["gpu_family"],
            "requested_ready_workers": int(row["requested_ready_workers"]),
            "provider_mutation_authorized_by_customer_request": False,
        }

    def release_scale_request(
        self,
        *,
        scale_request_id: str,
        scale_token: str,
        retryable: bool,
    ) -> dict[str, Any]:
        request_id = _bounded(scale_request_id, field="scale_request_id", maximum=64)
        token = _bounded(scale_token, field="scale_token", maximum=69)
        if not _SCALE_TOKEN.fullmatch(token):
            raise ValueError("scale_token_invalid")
        now = float(self._clock())
        next_status = "pending" if retryable else "cancelled"
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status, claim_token_digest, claim_expires_at FROM production_gpu_scale_requests "
                "WHERE scale_request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise WorkerPoolError("scale_request_unknown")
            if row["status"] != "claimed" or float(row["claim_expires_at"] or 0) <= now or not hmac.compare_digest(
                str(row["claim_token_digest"] or ""), _digest(token)
            ):
                raise WorkerLeaseConflict("active_scale_request_token_required")
            connection.execute(
                "UPDATE production_gpu_scale_requests SET status=?, claim_owner=NULL, "
                "claim_token_digest=NULL, claim_expires_at=NULL, updated_at=? WHERE scale_request_id=?",
                (next_status, now, request_id),
            )
        return {"scale_request_id": request_id, "status": next_status}

    def bind_customer_job(
        self,
        *,
        job_id: str,
        host_image_id: str,
        worker_image_ref: str,
        gpu_family: str,
        lease_seconds: float = 3600.0,
        heartbeat_ttl_seconds: float = 45.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        job = _job_id(job_id)
        host = _bounded(host_image_id, field="host_image_id", maximum=512)
        image = _image_ref(worker_image_ref)
        gpu = _bounded(gpu_family, field="gpu_family", maximum=128)
        lease = float(lease_seconds)
        heartbeat_ttl = float(heartbeat_ttl_seconds)
        if not 30.0 <= lease <= 86_400.0:
            raise ValueError("lease_seconds_out_of_range")
        if not 5.0 <= heartbeat_ttl <= 600.0:
            raise ValueError("heartbeat_ttl_seconds_out_of_range")
        fingerprint = release_fingerprint(host_image_id=host, worker_image_ref=image, gpu_family=gpu)
        now = float(self._clock())
        token = f"gwl_{secrets.token_hex(32)}"
        with self._transaction() as connection:
            self._expire_stale(connection, now=now, heartbeat_ttl_seconds=heartbeat_ttl)
            existing = connection.execute(
                "SELECT * FROM production_gpu_workers WHERE lease_job_id=?",
                (job,),
            ).fetchone()
            if existing is not None:
                raise WorkerLeaseConflict("job_already_has_active_worker_lease")
            row = connection.execute(
                "SELECT * FROM production_gpu_workers WHERE release_fingerprint=? AND state='ready' "
                "ORDER BY heartbeat_at DESC, registered_at, worker_id LIMIT 1",
                (fingerprint,),
            ).fetchone()
            if row is None:
                scale_id = self._scale_request(
                    connection,
                    fingerprint=fingerprint,
                    host_image_id=host,
                    worker_image_ref=image,
                    gpu_family=gpu,
                    requested_ready_workers=1,
                    reason="customer_waiting_no_ready_worker",
                    now=now,
                )
                return {
                    "schema_version": PRODUCTION_GPU_BINDING_SCHEMA_VERSION,
                    "job_id": job,
                    "status": "queued_waiting_for_warm_worker",
                    "release_fingerprint": fingerprint,
                    "scale_request_id": scale_id,
                    "customer_request_provider_calls": 0,
                    "cold_provisioning_started_in_request_path": False,
                    "bind_latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                }
            worker = str(row["worker_id"])
            updated = connection.execute(
                "UPDATE production_gpu_workers SET state='leased', lease_job_id=?, "
                "lease_token_digest=?, lease_expires_at=?, transition_reason='customer_bound' "
                "WHERE worker_id=? AND state='ready'",
                (job, _digest(token), now + lease, worker),
            )
            if updated.rowcount != 1:
                raise WorkerLeaseConflict("atomic_warm_worker_bind_lost")
        return {
            "schema_version": PRODUCTION_GPU_BINDING_SCHEMA_VERSION,
            "job_id": job,
            "status": "bound_to_ready_worker",
            "worker_id": worker,
            "endpoint_ref": row["endpoint_ref"],
            "provider": row["provider"],
            "release_fingerprint": fingerprint,
            "lease_token": token,
            "lease_expires_at_epoch": now + lease,
            "customer_request_provider_calls": 0,
            "cold_provisioning_started_in_request_path": False,
            "bind_latency_ms": round((time.monotonic() - started) * 1000.0, 3),
        }

    def release_worker(
        self,
        *,
        worker_id: str,
        job_id: str,
        lease_token: str,
        healthy: bool,
    ) -> dict[str, Any]:
        worker = _worker_id(worker_id)
        job = _job_id(job_id)
        token = _lease_token(lease_token)
        now = float(self._clock())
        next_state = "ready" if healthy else "quarantined"
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state, lease_job_id, lease_token_digest FROM production_gpu_workers WHERE worker_id=?",
                (worker,),
            ).fetchone()
            if row is None:
                raise UnknownWorker("worker_unknown")
            if row["state"] != "leased" or row["lease_job_id"] != job or not hmac.compare_digest(
                str(row["lease_token_digest"] or ""), _digest(token)
            ):
                raise WorkerLeaseConflict("active_worker_job_lease_required")
            connection.execute(
                "UPDATE production_gpu_workers SET state=?, heartbeat_at=?, lease_job_id=NULL, "
                "lease_token_digest=NULL, lease_expires_at=NULL, transition_reason=? WHERE worker_id=?",
                (next_state, now, "customer_job_completed" if healthy else "customer_job_worker_unhealthy", worker),
            )
        return {"worker_id": worker, "state": next_state, "ready_for_customer_binding": healthy}

    def reconcile_min_ready(
        self,
        *,
        host_image_id: str,
        worker_image_ref: str,
        gpu_family: str,
        min_ready_workers: int,
        heartbeat_ttl_seconds: float = 45.0,
    ) -> dict[str, Any]:
        host = _bounded(host_image_id, field="host_image_id", maximum=512)
        image = _image_ref(worker_image_ref)
        gpu = _bounded(gpu_family, field="gpu_family", maximum=128)
        target = int(min_ready_workers)
        if not 1 <= target <= 100:
            raise ValueError("min_ready_workers_out_of_range")
        now = float(self._clock())
        fingerprint = release_fingerprint(host_image_id=host, worker_image_ref=image, gpu_family=gpu)
        with self._transaction() as connection:
            self._expire_stale(connection, now=now, heartbeat_ttl_seconds=float(heartbeat_ttl_seconds))
            ready = int(connection.execute(
                "SELECT COUNT(*) FROM production_gpu_workers WHERE release_fingerprint=? AND state='ready'",
                (fingerprint,),
            ).fetchone()[0])
            deficit = max(0, target - ready)
            scale_id = None
            if deficit:
                scale_id = self._scale_request(
                    connection, fingerprint=fingerprint, host_image_id=host,
                    worker_image_ref=image, gpu_family=gpu,
                    requested_ready_workers=deficit, reason="min_ready_capacity_deficit", now=now,
                )
        return {
            "schema_version": PRODUCTION_GPU_SCALE_REQUEST_SCHEMA_VERSION,
            "release_fingerprint": fingerprint,
            "min_ready_workers": target,
            "ready_workers": ready,
            "deficit": deficit,
            "scale_request_id": scale_id,
            "provider_calls_performed": 0,
            "autoscaler_must_process_asynchronously": True,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            worker_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM production_gpu_workers GROUP BY state"
            ).fetchall()
            pending = int(connection.execute(
                "SELECT COUNT(*) FROM production_gpu_scale_requests WHERE status IN ('pending','claimed')"
            ).fetchone()[0])
        counts = {"ready": 0, "leased": 0, "draining": 0, "quarantined": 0}
        counts.update({str(row["state"]): int(row["count"]) for row in worker_rows})
        return {
            "schema_version": PRODUCTION_GPU_WORKER_POOL_SCHEMA_VERSION,
            "worker_counts": counts,
            "pending_scale_requests": pending,
        }


def build_production_startup_readiness(
    *,
    host_image_id: str,
    worker_image_ref: str,
    gpu_family: str,
    min_ready_workers: int,
    bind_slo_seconds: float,
    cold_replenishment_slo_seconds: float,
    live_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed release gate without upgrading local proof to live proof."""

    blockers: list[str] = []
    try:
        fingerprint = release_fingerprint(
            host_image_id=host_image_id, worker_image_ref=worker_image_ref, gpu_family=gpu_family
        )
    except ValueError as exc:
        fingerprint = ""
        blockers.append(str(exc))
    if int(min_ready_workers) < 1:
        blockers.append("min_ready_workers_must_be_positive")
    if not 0 < float(bind_slo_seconds) <= 30:
        blockers.append("warm_bind_slo_must_be_at_most_30_seconds")
    if not 0 < float(cold_replenishment_slo_seconds) <= 1800:
        blockers.append("cold_replenishment_slo_must_be_at_most_1800_seconds")
    evidence = dict(live_evidence or {})
    required_live = {
        "exact_release_fingerprint": evidence.get("release_fingerprint") == fingerprint and bool(fingerprint),
        "baked_host_image": evidence.get("baked_host_image_verified") is True,
        "worker_image_cache": evidence.get("worker_image_cached_on_host_verified") is True,
        "warm_stack_readiness": evidence.get("all_required_ready_checks_observed") is True,
        "ready_worker_count": int(evidence.get("ready_worker_count") or 0) >= int(min_ready_workers),
        "current_warm_capacity": evidence.get("current_capacity_deployed") is True
        and int(evidence.get("current_provider_live_worker_count") or 0)
        >= int(min_ready_workers),
        "warm_bind_p95": float(evidence.get("warm_bind_p95_seconds") or 1e12) <= float(bind_slo_seconds),
        "cold_replenishment_p95": float(evidence.get("cold_replenishment_p95_seconds") or 1e12)
        <= float(cold_replenishment_slo_seconds),
        "async_scale_path": evidence.get("customer_request_provider_calls") == 0
        and evidence.get("async_scale_replenishment_proven") is True,
        "rollback": evidence.get("rollback_drill_passed") is True,
        "provider_inventory": evidence.get("provider_inventory_confirmed") is True,
        "teardown": evidence.get("teardown_and_absence_confirmed") is True,
    }
    blockers.extend(f"live_evidence_missing:{name}" for name, passed in required_live.items() if not passed)
    return {
        "schema_version": PRODUCTION_GPU_STARTUP_READINESS_SCHEMA_VERSION,
        "status": "customer_launch_ready" if not blockers else "local_contract_ready_live_proof_required",
        "release_fingerprint": fingerprint or None,
        "startup_targets": {
            "min_ready_workers": int(min_ready_workers),
            "warm_bind_slo_seconds": float(bind_slo_seconds),
            "cold_replenishment_slo_seconds": float(cold_replenishment_slo_seconds),
            "customer_request_cold_provisioning_allowed": False,
            "cold_replenishment_is_outside_customer_request_slo": True,
        },
        "live_evidence_checks": required_live,
        "blockers": blockers,
        "claim_boundary": {
            "local_registry_tests_are_not_live_gpu_startup_proof": True,
            "host_image_build_is_not_warm_pool_capacity_proof": True,
            "campaign_cold_start_is_release_engineering_evidence_only": True,
            "customer_launch_claim_requires_exact_release_live_evidence": True,
        },
    }


def _read_token(explicit_token: str | None) -> str:
    if explicit_token is not None:
        token = explicit_token.strip()
    else:
        path_text = os.getenv(POOL_TOKEN_FILE_ENV, "").strip()
        if not path_text:
            raise RuntimeError("production_gpu_pool_token_file_required")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("production_gpu_pool_token_file_invalid")
        if path.stat().st_mode & 0o077:
            raise RuntimeError("production_gpu_pool_token_file_permissions_too_open")
        token = path.read_text(encoding="utf-8").strip()
    if len(token.encode("utf-8")) < 32:
        raise RuntimeError("production_gpu_pool_token_too_short")
    return token


def create_production_gpu_worker_pool_app(
    *, database_path: str | Path, auth_token: str | None = None, clock: Callable[[], float] = time.time
) -> Any:
    """Create the private authenticated pool API."""

    from fastapi import Depends, FastAPI, Header, HTTPException, status

    pool = ProductionGpuWorkerPool(database_path, clock=clock)
    expected_token = _read_token(auth_token)

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    app = FastAPI(title="Blueprint Production GPU Worker Pool", version=PRODUCTION_GPU_WORKER_POOL_SCHEMA_VERSION)
    app.state.pool = pool

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok", **pool.snapshot()}

    @app.post("/v1/workers/ready", dependencies=[Depends(require_auth)])
    def ready(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.register_ready_worker(
                worker_id=str(payload.get("worker_id") or ""),
                provider=str(payload.get("provider") or ""),
                host_image_id=str(payload.get("host_image_id") or ""),
                worker_image_ref=str(payload.get("worker_image_ref") or ""),
                gpu_family=str(payload.get("gpu_family") or ""),
                endpoint_ref=str(payload.get("endpoint_ref") or ""),
                readiness=dict(payload.get("readiness") or {}),
            )
        except (ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/workers/{worker_id}/heartbeat", dependencies=[Depends(require_auth)])
    def heartbeat(worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.heartbeat(
                worker_id=worker_id,
                lease_token=payload.get("lease_token"),
                # The private infrastructure token protecting this route is
                # never issued to customer lease holders.
                trusted_worker_agent=True,
            )
        except (ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/workers/{worker_id}/quarantine", dependencies=[Depends(require_auth)])
    def quarantine(worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.quarantine_worker(
                worker_id=worker_id,
                reason=str(payload.get("reason") or "provider_teardown"),
            )
        except (ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/workers/{worker_id}/release", dependencies=[Depends(require_auth)])
    def release(worker_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.release_worker(
                worker_id=worker_id,
                job_id=str(payload.get("job_id") or ""),
                lease_token=str(payload.get("lease_token") or ""),
                healthy=payload.get("healthy") is True,
            )
        except (ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/customer-jobs/bind", dependencies=[Depends(require_auth)])
    def bind(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.bind_customer_job(
                job_id=str(payload.get("job_id") or ""),
                host_image_id=str(payload.get("host_image_id") or ""),
                worker_image_ref=str(payload.get("worker_image_ref") or ""),
                gpu_family=str(payload.get("gpu_family") or ""),
                lease_seconds=float(payload.get("lease_seconds") or 3600),
            )
        except (TypeError, ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/autoscaler/scale-requests/claim", dependencies=[Depends(require_auth)])
    def claim_scale(payload: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return pool.claim_scale_request(
                autoscaler_id=str(payload.get("autoscaler_id") or ""),
                lease_seconds=float(payload.get("lease_seconds") or 120),
            )
        except (TypeError, ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post("/v1/autoscaler/reconcile", dependencies=[Depends(require_auth)])
    def reconcile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.reconcile_min_ready(
                host_image_id=str(payload.get("host_image_id") or ""),
                worker_image_ref=str(payload.get("worker_image_ref") or ""),
                gpu_family=str(payload.get("gpu_family") or ""),
                min_ready_workers=int(payload.get("min_ready_workers") or 0),
                heartbeat_ttl_seconds=float(payload.get("heartbeat_ttl_seconds") or 45),
            )
        except (TypeError, ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    @app.post(
        "/v1/autoscaler/scale-requests/{scale_request_id}/release",
        dependencies=[Depends(require_auth)],
    )
    def release_scale(scale_request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return pool.release_scale_request(
                scale_request_id=scale_request_id,
                scale_token=str(payload.get("scale_token") or ""),
                retryable=payload.get("retryable") is True,
            )
        except (ValueError, WorkerPoolError) as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc

    return app


def readiness_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed production GPU startup promotion gate."
    )
    parser.add_argument("--host-image-id", required=True)
    parser.add_argument("--worker-image-ref", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--min-ready-workers", type=int, default=1)
    parser.add_argument("--bind-slo-seconds", type=float, default=10.0)
    parser.add_argument("--cold-replenishment-slo-seconds", type=float, default=180.0)
    parser.add_argument("--live-evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    live_evidence: Mapping[str, Any] | None = None
    if args.live_evidence:
        payload = read_json_any(Path(args.live_evidence).expanduser().resolve())
        if not isinstance(payload, Mapping):
            raise SystemExit("live evidence must be a JSON object")
        live_evidence = payload
    result = build_production_startup_readiness(
        host_image_id=args.host_image_id,
        worker_image_ref=args.worker_image_ref,
        gpu_family=args.gpu_family,
        min_ready_workers=args.min_ready_workers,
        bind_slo_seconds=args.bind_slo_seconds,
        cold_replenishment_slo_seconds=args.cold_replenishment_slo_seconds,
        live_evidence=live_evidence,
    )
    write_json(Path(args.output).expanduser().resolve(), result)
    print(f"[production-gpu-startup] status={result['status']}")
    if result["blockers"]:
        print("[production-gpu-startup] blockers=" + ",".join(result["blockers"]))
    return 0 if result["status"] == "customer_launch_ready" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("Public binding is disabled; put the pool behind private IAM.")
    import uvicorn

    uvicorn.run(create_production_gpu_worker_pool_app(database_path=args.database), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
