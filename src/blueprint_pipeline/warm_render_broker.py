"""Durable broker for warm-render jobs.

The broker is deliberately small and provider-neutral. SQLite supplies the
transactional queue for a single control-plane deployment; production can put
the FastAPI surface behind private IAM or a service mesh without changing the
worker/client contract. Caller labels never become filesystem paths.
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

WARM_RENDER_BROKER_SCHEMA_VERSION = "warm_render_broker.v1"
WARM_RENDER_JOB_SCHEMA_VERSION = "warm_render_broker_job.v1"
WARM_RENDER_RESULT_SCHEMA_VERSION = "warm_render_broker_result.v1"
WARM_RENDER_BROKER_TOKEN_FILE_ENV = "BLUEPRINT_WARM_RENDER_BROKER_TOKEN_FILE"
MAX_BODY_BYTES = 1_048_576
MAX_JSON_DEPTH = 32
DEFAULT_LEASE_SECONDS = 900.0
_CANONICAL_JOB_ID = re.compile(r"\Awrj_[0-9a-f]{32}\Z")
_CLAIM_TOKEN = re.compile(r"\Awrc_[0-9a-f]{64}\Z")
_WORKER_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class WarmRenderBrokerError(RuntimeError):
    """Base error for durable broker contract violations."""


class IdempotencyConflict(WarmRenderBrokerError):
    """An idempotency key was reused for different request content."""


class ClaimConflict(WarmRenderBrokerError):
    """A result was committed without the active lease token."""


class UnknownWarmRenderJob(WarmRenderBrokerError):
    """The canonical server job ID does not exist."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload_not_canonical_json") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text(value: Any, *, field: str, maximum: int, required: bool) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if required and not text:
        raise ValueError(f"{field}_required")
    if len(text.encode("utf-8")) > maximum:
        raise ValueError(f"{field}_too_large")
    return text


def _canonical_job_id(value: Any) -> str:
    text = _bounded_text(value, field="canonical_job_id", maximum=36, required=True)
    if not _CANONICAL_JOB_ID.fullmatch(text):
        raise ValueError("canonical_job_id_invalid")
    return text


def _claim_token(value: Any) -> str:
    text = _bounded_text(value, field="claim_token", maximum=68, required=True)
    if not _CLAIM_TOKEN.fullmatch(text):
        raise ValueError("claim_token_invalid")
    return text


def _json_depth(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_JSON_DEPTH:
        return depth
    if isinstance(value, Mapping):
        return max(
            [depth, *(_json_depth(item, depth=depth + 1) for item in value.values())]
        )
    if isinstance(value, list):
        return max([depth, *(_json_depth(item, depth=depth + 1) for item in value)])
    return depth


class DurableWarmRenderQueue:
    """Transactional queue with restart-safe leases and idempotent commits."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS warm_render_jobs (
                    canonical_job_id TEXT PRIMARY KEY,
                    server_idempotency_key TEXT NOT NULL UNIQUE,
                    request_fingerprint TEXT NOT NULL,
                    client_request_label TEXT NOT NULL,
                    scenario_json TEXT NOT NULL,
                    session_nonce TEXT NOT NULL,
                    stop_requested INTEGER NOT NULL CHECK (stop_requested IN (0, 1)),
                    status TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'completed')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_token_digest TEXT,
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    result_fingerprint TEXT,
                    completed_claim_token_digest TEXT
                );
                CREATE INDEX IF NOT EXISTS warm_render_jobs_claim_order
                    ON warm_render_jobs(status, lease_expires_at, created_at, canonical_job_id);
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

    def submit(
        self,
        *,
        scenario: Mapping[str, Any],
        idempotency_key: str,
        client_request_label: str = "",
        session_nonce: str = "",
        stop: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario_must_be_mapping")
        supplied_key = _bounded_text(
            idempotency_key,
            field="idempotency_key",
            maximum=256,
            required=True,
        )
        label = _bounded_text(
            client_request_label,
            field="client_request_label",
            maximum=256,
            required=False,
        )
        nonce = _bounded_text(
            session_nonce,
            field="session_nonce",
            maximum=256,
            required=False,
        )
        scenario_json = _canonical_json(dict(scenario))
        if len(scenario_json.encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("scenario_too_large")
        request_fingerprint = _sha256_text(
            _canonical_json(
                {
                    "client_request_label": label,
                    "scenario": dict(scenario),
                    "session_nonce": nonce,
                    "stop": bool(stop),
                }
            )
        )
        server_idempotency_key = f"wri_{_sha256_text(supplied_key)}"
        now = float(self._clock())
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT canonical_job_id, request_fingerprint, status
                  FROM warm_render_jobs
                 WHERE server_idempotency_key = ?
                """,
                (server_idempotency_key,),
            ).fetchone()
            if existing is not None:
                if not hmac.compare_digest(
                    str(existing["request_fingerprint"]), request_fingerprint
                ):
                    raise IdempotencyConflict("idempotency_key_payload_mismatch")
                return {
                    "schema_version": WARM_RENDER_JOB_SCHEMA_VERSION,
                    "canonical_job_id": existing["canonical_job_id"],
                    "server_idempotency_key": server_idempotency_key,
                    "status": existing["status"],
                    "idempotent_replay": True,
                }
            canonical_id = f"wrj_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO warm_render_jobs (
                    canonical_job_id, server_idempotency_key, request_fingerprint,
                    client_request_label, scenario_json, session_nonce,
                    stop_requested, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    canonical_id,
                    server_idempotency_key,
                    request_fingerprint,
                    label,
                    scenario_json,
                    nonce,
                    1 if stop else 0,
                    now,
                    now,
                ),
            )
        return {
            "schema_version": WARM_RENDER_JOB_SCHEMA_VERSION,
            "canonical_job_id": canonical_id,
            "server_idempotency_key": server_idempotency_key,
            "status": "queued",
            "idempotent_replay": False,
        }

    def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> dict[str, Any] | None:
        owner = _bounded_text(worker_id, field="worker_id", maximum=128, required=True)
        if not _WORKER_ID.fullmatch(owner):
            raise ValueError("worker_id_invalid")
        lease = float(lease_seconds)
        if not 1.0 <= lease <= 86_400.0:
            raise ValueError("lease_seconds_out_of_range")
        now = float(self._clock())
        token = f"wrc_{secrets.token_hex(32)}"
        token_digest = _sha256_text(token)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT *
                  FROM warm_render_jobs
                 WHERE status = 'queued'
                    OR (status = 'leased' AND lease_expires_at <= ?)
                 ORDER BY created_at, canonical_job_id
                 LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            canonical_id = str(row["canonical_job_id"])
            updated = connection.execute(
                """
                UPDATE warm_render_jobs
                   SET status = 'leased', lease_owner = ?, lease_token_digest = ?,
                       lease_expires_at = ?, updated_at = ?, attempt_count = attempt_count + 1
                 WHERE canonical_job_id = ?
                   AND (status = 'queued' OR (status = 'leased' AND lease_expires_at <= ?))
                """,
                (owner, token_digest, now + lease, now, canonical_id, now),
            )
            if updated.rowcount != 1:
                raise ClaimConflict("atomic_job_claim_lost")
            claimed = connection.execute(
                "SELECT * FROM warm_render_jobs WHERE canonical_job_id = ?",
                (canonical_id,),
            ).fetchone()
        assert claimed is not None
        return {
            "schema_version": WARM_RENDER_JOB_SCHEMA_VERSION,
            "canonical_job_id": canonical_id,
            "server_idempotency_key": claimed["server_idempotency_key"],
            "client_request_label": claimed["client_request_label"],
            "scenario": json.loads(str(claimed["scenario_json"])),
            "session_nonce": claimed["session_nonce"],
            "stop": bool(claimed["stop_requested"]),
            "claim_token": token,
            "lease_expires_at": now + lease,
            "attempt_count": int(claimed["attempt_count"]),
        }

    def publish_result(
        self,
        *,
        canonical_job_id: str,
        claim_token: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        canonical_id = _canonical_job_id(canonical_job_id)
        token = _claim_token(claim_token)
        if not isinstance(result, Mapping):
            raise ValueError("result_must_be_mapping")
        committed_result = dict(result)
        committed_result["canonical_job_id"] = canonical_id
        result_json = _canonical_json(committed_result)
        if len(result_json.encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("result_too_large")
        result_fingerprint = _sha256_text(result_json)
        token_digest = _sha256_text(token)
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM warm_render_jobs WHERE canonical_job_id = ?",
                (canonical_id,),
            ).fetchone()
            if row is None:
                raise UnknownWarmRenderJob("canonical_job_id_unknown")
            if row["status"] == "completed":
                same_token = hmac.compare_digest(
                    str(row["completed_claim_token_digest"] or ""), token_digest
                )
                same_result = hmac.compare_digest(
                    str(row["result_fingerprint"] or ""), result_fingerprint
                )
                if not (same_token and same_result):
                    raise ClaimConflict("completed_result_conflict")
                return {
                    "schema_version": WARM_RENDER_RESULT_SCHEMA_VERSION,
                    "canonical_job_id": canonical_id,
                    "status": "completed",
                    "idempotent_replay": True,
                    "result_fingerprint": result_fingerprint,
                }
            active_token = str(row["lease_token_digest"] or "")
            if row["status"] != "leased" or not hmac.compare_digest(
                active_token, token_digest
            ):
                raise ClaimConflict("active_claim_token_required")
            if float(row["lease_expires_at"] or 0.0) <= now:
                raise ClaimConflict("claim_lease_expired")
            connection.execute(
                """
                UPDATE warm_render_jobs
                   SET status = 'completed', result_json = ?, result_fingerprint = ?,
                       completed_claim_token_digest = ?, updated_at = ?,
                       lease_owner = NULL, lease_token_digest = NULL, lease_expires_at = NULL
                 WHERE canonical_job_id = ?
                """,
                (result_json, result_fingerprint, token_digest, now, canonical_id),
            )
        return {
            "schema_version": WARM_RENDER_RESULT_SCHEMA_VERSION,
            "canonical_job_id": canonical_id,
            "status": "completed",
            "idempotent_replay": False,
            "result_fingerprint": result_fingerprint,
        }

    def get_result(
        self,
        *,
        canonical_job_id: str,
        session_nonce: str | None = None,
    ) -> dict[str, Any] | None:
        canonical_id = _canonical_job_id(canonical_job_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, session_nonce, result_json, result_fingerprint
                  FROM warm_render_jobs
                 WHERE canonical_job_id = ?
                """,
                (canonical_id,),
            ).fetchone()
        if row is None:
            raise UnknownWarmRenderJob("canonical_job_id_unknown")
        expected_nonce = str(row["session_nonce"] or "")
        if expected_nonce:
            supplied_nonce = _bounded_text(
                session_nonce,
                field="session_nonce",
                maximum=256,
                required=True,
            )
            if not hmac.compare_digest(expected_nonce, supplied_nonce):
                raise ClaimConflict("session_nonce_mismatch")
        if row["status"] != "completed" or not row["result_json"]:
            return None
        return {
            "schema_version": WARM_RENDER_RESULT_SCHEMA_VERSION,
            "canonical_job_id": canonical_id,
            "status": "completed",
            "result_fingerprint": row["result_fingerprint"],
            "result": json.loads(str(row["result_json"])),
        }

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM warm_render_jobs GROUP BY status"
            ).fetchall()
        counts = {"queued": 0, "leased": 0, "completed": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        return counts


def _read_auth_token(explicit_token: str | None) -> str:
    if explicit_token is not None:
        token = explicit_token.strip()
    else:
        path_text = os.getenv(WARM_RENDER_BROKER_TOKEN_FILE_ENV, "").strip()
        if not path_text:
            raise RuntimeError("warm_render_broker_token_file_required")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("warm_render_broker_token_file_invalid")
        if path.stat().st_mode & 0o077:
            raise RuntimeError("warm_render_broker_token_file_permissions_too_open")
        token = path.read_text(encoding="utf-8").strip()
    if len(token.encode("utf-8")) < 32:
        raise RuntimeError("warm_render_broker_token_too_short")
    return token


async def _request_json(request: Any) -> dict[str, Any]:
    from fastapi import HTTPException, status

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from exc
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST) from exc
    if not isinstance(payload, Mapping):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    if _json_depth(payload) > MAX_JSON_DEPTH:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return dict(payload)


def create_warm_render_broker_app(
    *,
    database_path: str | Path,
    auth_token: str | None = None,
    clock: Callable[[], float] = time.time,
) -> Any:
    """Create the authenticated durable warm-render broker service."""

    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status

    # FastAPI resolves postponed endpoint annotations against module globals.
    globals()["Request"] = Request
    globals()["Response"] = Response

    queue = DurableWarmRenderQueue(database_path, clock=clock)
    expected_token = _read_auth_token(auth_token)

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    app = FastAPI(title="Blueprint Warm Render Broker", version=WARM_RENDER_BROKER_SCHEMA_VERSION)
    app.state.queue = queue

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": WARM_RENDER_BROKER_SCHEMA_VERSION,
            "queue_counts": queue.counts(),
        }

    @app.post("/v1/warm-render/jobs", dependencies=[Depends(require_auth)])
    async def submit_job(request: Request) -> dict[str, Any]:
        payload = await _request_json(request)
        try:
            return queue.submit(
                scenario=dict(payload.get("scenario") or {}),
                idempotency_key=str(payload.get("idempotency_key") or ""),
                client_request_label=str(payload.get("client_request_label") or ""),
                session_nonce=str(payload.get("session_nonce") or ""),
                stop=payload.get("stop") is True,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT) from exc

    @app.post(
        "/v1/warm-render/jobs/claim",
        dependencies=[Depends(require_auth)],
        response_model=None,
    )
    async def claim_job(request: Request) -> Response | dict[str, Any]:
        payload = await _request_json(request)
        try:
            claimed = queue.claim(
                worker_id=str(payload.get("worker_id") or ""),
                lease_seconds=float(payload.get("lease_seconds") or DEFAULT_LEASE_SECONDS),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT) from exc
        return claimed if claimed is not None else Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.put(
        "/v1/warm-render/jobs/{canonical_job_id}/result",
        dependencies=[Depends(require_auth)],
    )
    async def publish_result(
        canonical_job_id: str,
        request: Request,
    ) -> dict[str, Any]:
        payload = await _request_json(request)
        try:
            return queue.publish_result(
                canonical_job_id=canonical_job_id,
                claim_token=str(payload.get("claim_token") or ""),
                result=dict(payload.get("result") or {}),
            )
        except UnknownWarmRenderJob as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except ClaimConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT) from exc

    @app.get(
        "/v1/warm-render/jobs/{canonical_job_id}/result",
        dependencies=[Depends(require_auth)],
        response_model=None,
    )
    def get_result(
        canonical_job_id: str,
        x_warm_session_nonce: str | None = Header(default=None),
    ) -> Response | dict[str, Any]:
        try:
            result = queue.get_result(
                canonical_job_id=canonical_job_id,
                session_nonce=x_warm_session_nonce,
            )
        except UnknownWarmRenderJob as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc
        except (ClaimConflict, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN) from exc
        return result if result is not None else Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("Public binding is disabled; put the broker behind private IAM.")
    import uvicorn

    app = create_warm_render_broker_app(database_path=args.database)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
