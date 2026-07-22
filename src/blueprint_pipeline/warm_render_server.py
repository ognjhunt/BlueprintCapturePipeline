"""Persistent warm-render serve loop: keep Isaac + the scene loaded ONCE and serve a stream of task
render jobs, so each rerun skips image pull + Isaac boot + stage load + most settle.

Design: the Isaac setup/render are INJECTED (``render_one``) — this module imports NO isaacsim and NO
pxr, so the control flow (poll, render, publish, stop / idle-timeout / max-jobs, error isolation) is
hermetically testable. The GPU runner wires in the real "boot once, render one scenario" function for
the (guarded) on-GPU validation. Jobs/results flow through a swappable :class:`JobSource` backend so
the same loop runs against a local directory (tests / shared volume) or signed object-store URLs.
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .core.security_controls import (
    exact_https_origin,
    fetch_bounded_https,
    fetch_bounded_service_url,
    origins_from_env,
)


WARM_SIGNED_URL_ALLOWED_ORIGINS_ENV = (
    "BLUEPRINT_WARM_SIGNED_URL_ALLOWED_ORIGINS"
)
WARM_BROKER_ALLOWED_ORIGINS_ENV = "BLUEPRINT_WARM_BROKER_ALLOWED_ORIGINS"
WARM_SINGLE_OBJECT_MAX_BYTES = 16 * 1024 * 1024
WARM_BROKER_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class PresignedUrlAccessError(RuntimeError):
    """Classified presigned URL access failure.

    Carries only status/classification metadata; it intentionally never includes the raw URL.
    """

    def __init__(self, *, operation: str, status_code: int, classification: str) -> None:
        super().__init__(f"{classification}:{operation}:http_{status_code}")
        self.operation = operation
        self.status_code = int(status_code)
        self.classification = classification


class WarmInboxUnrecoverable(RuntimeError):
    """Raised when the warm inbox has repeated hard failures and should stop polling."""

    def __init__(self, *, reason: str, failures: int) -> None:
        super().__init__(f"{reason}:consecutive_failures={failures}")
        self.reason = reason
        self.failures = int(failures)


class WarmBrokerContractError(RuntimeError):
    """The durable broker returned a malformed or unauthorized response."""


def _http_error_classification(code: int) -> str:
    if int(code) in (401, 403):
        return "presigned_url_expired_or_forbidden"
    return f"presigned_url_http_error_{int(code)}"


@dataclass
class WarmJob:
    """One render request handed to the warm worker. ``stop=True`` is the shutdown sentinel."""

    request_id: str
    scenario: dict[str, Any] = field(default_factory=dict)
    stop: bool = False
    session_nonce: str = ""
    client_request_label: str = ""
    server_idempotency_key: str = ""


class JobSource(Protocol):
    """Swappable transport for warm jobs/results (local dir, signed URLs, ...)."""

    def poll(self) -> Optional[WarmJob]:
        """Return the next claimed job, or ``None`` if the queue is currently empty (non-blocking)."""

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        """Record the render result for ``request_id`` so the submitter can collect it."""


def serve_render_loop(
    *,
    render_one: Callable[[dict[str, Any]], dict[str, Any]],
    job_source: JobSource,
    idle_timeout_s: float = 600.0,
    max_jobs: Optional[int] = None,
    poll_interval_s: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Serve warm render jobs until a stop sentinel, idle timeout, or ``max_jobs`` is reached.

    ``render_one(scenario) -> result`` is the injected (Isaac-bound, on GPU) renderer; a raising
    ``render_one`` is isolated to that job (recorded as an error result) and never kills the loop, so
    one bad task can't waste the warm pod. Returns ``{jobs_served, exit_reason}``.
    """
    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    served = 0
    last_activity = clock()
    while True:
        if max_jobs is not None and served >= max_jobs:
            _log(f"warm serve loop: max_jobs={max_jobs} reached after {served} job(s)")
            return {"jobs_served": served, "exit_reason": "max_jobs"}

        try:
            job = job_source.poll()
        except PresignedUrlAccessError as exc:
            _log(f"warm serve loop: inbox access failed: {exc.classification}")
            return {
                "jobs_served": served,
                "exit_reason": "inbox_unrecoverable",
                "blocker": exc.classification,
            }
        except WarmInboxUnrecoverable as exc:
            _log(f"warm serve loop: inbox unrecoverable: {exc.reason}")
            return {
                "jobs_served": served,
                "exit_reason": "inbox_unrecoverable",
                "blocker": exc.reason,
                "consecutive_failures": exc.failures,
            }
        if job is None:
            if clock() - last_activity >= idle_timeout_s:
                _log(f"warm serve loop: idle {idle_timeout_s}s elapsed; exiting after {served} job(s)")
                return {"jobs_served": served, "exit_reason": "idle_timeout"}
            sleep(poll_interval_s)
            continue

        if job.stop:
            stop_result = {
                "status": "stopped",
                "request_id": job.request_id,
                "canonical_job_id": job.request_id,
                "stop_acknowledged": True,
            }
            if job.session_nonce:
                stop_result["warm_session_nonce"] = job.session_nonce
            job_source.publish_result(job.request_id, stop_result)
            _log(f"warm serve loop: stop sentinel received after {served} job(s)")
            return {"jobs_served": served, "exit_reason": "stop_requested"}

        _log(f"warm serve loop: rendering request_id={job.request_id}")
        try:
            result = render_one(job.scenario)
            if not isinstance(result, dict):
                result = {"status": "ok", "result": result}
            result.setdefault("status", "ok")
        except Exception as exc:  # noqa: BLE001 - isolate one job's failure from the warm pod
            result = {"status": "error", "error": repr(exc)}
            _log(f"warm serve loop: request_id={job.request_id} render error: {exc!r}")

        result["request_id"] = job.request_id
        result["canonical_job_id"] = job.request_id
        if job.client_request_label:
            result["client_request_label"] = job.client_request_label
        if job.server_idempotency_key:
            result["server_idempotency_key"] = job.server_idempotency_key
        if job.session_nonce:
            result["warm_session_nonce"] = job.session_nonce
        job_source.publish_result(job.request_id, result)
        served += 1
        last_activity = clock()


class FileJobSource:
    """Local/shared-volume adapter over the same durable queue contract."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            from .warm_render_broker import DurableWarmRenderQueue
        except ImportError:  # flat worker bundle
            from warm_render_broker import DurableWarmRenderQueue  # type: ignore[no-redef]

        self._queue = DurableWarmRenderQueue(self.root / "warm_render_queue.sqlite3")
        self._worker_id = f"file-worker-{uuid.uuid4().hex}"
        self._active_claim_tokens: dict[str, str] = {}

    def submit(self, request_id: str, scenario: dict[str, Any]) -> str:
        submitted = self._queue.submit(
            scenario=scenario,
            idempotency_key=f"file-job:{request_id}",
            client_request_label=request_id,
        )
        return str(submitted["canonical_job_id"])

    def submit_stop(self, *, idempotency_key: str | None = None) -> str:
        submitted = self._queue.submit(
            scenario={},
            idempotency_key=idempotency_key or f"file-stop:{uuid.uuid4().hex}",
            client_request_label="stop",
            stop=True,
        )
        return str(submitted["canonical_job_id"])

    def poll(self) -> Optional[WarmJob]:
        payload = self._queue.claim(
            worker_id=self._worker_id,
            lease_seconds=900,
        )
        if payload is None:
            return None
        canonical_id = str(payload["canonical_job_id"])
        self._active_claim_tokens[canonical_id] = str(payload["claim_token"])
        return WarmJob(
            request_id=canonical_id,
            scenario=dict(payload.get("scenario") or {}),
            stop=bool(payload.get("stop")),
            session_nonce=str(payload.get("session_nonce") or ""),
            client_request_label=str(payload.get("client_request_label") or ""),
            server_idempotency_key=str(payload.get("server_idempotency_key") or ""),
        )

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        token = self._active_claim_tokens.get(request_id)
        if token is None:
            raise WarmBrokerContractError("local_warm_queue_active_claim_missing")
        self._queue.publish_result(
            canonical_job_id=request_id,
            claim_token=token,
            result=result,
        )
        self._active_claim_tokens.pop(request_id, None)

    def collect_result(self, request_id: str) -> Optional[dict[str, Any]]:
        payload = self._queue.get_result(canonical_job_id=request_id)
        return dict(payload["result"]) if payload is not None else None


def _http_get_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    allowed_origins = origins_from_env(WARM_SIGNED_URL_ALLOWED_ORIGINS_ENV)
    if not allowed_origins:
        allowed_origins = (exact_https_origin(url),)
    return fetch_bounded_https(
        url,
        timeout_seconds=max(1, min(int(timeout), 600)),
        max_bytes=WARM_SINGLE_OBJECT_MAX_BYTES,
        allowed_origins=allowed_origins,
        max_redirects=0,
    ).body


def _http_put_bytes(url: str, data: bytes, *, timeout: float = 60.0) -> bytes:
    if len(data) > WARM_SINGLE_OBJECT_MAX_BYTES:
        raise WarmBrokerContractError("warm_single_object_payload_too_large")
    allowed_origins = origins_from_env(WARM_SIGNED_URL_ALLOWED_ORIGINS_ENV)
    if not allowed_origins:
        allowed_origins = (exact_https_origin(url),)
    return fetch_bounded_https(
        url,
        method="PUT",
        data=data,
        headers={"Content-Type": "application/json"},
        timeout_seconds=max(1, min(int(timeout), 600)),
        max_bytes=1024 * 1024,
        allowed_origins=allowed_origins,
        max_redirects=0,
    ).body


_CANONICAL_BROKER_JOB_ID = re.compile(r"\Awrj_[0-9a-f]{32}\Z")
_BROKER_CLAIM_TOKEN = re.compile(r"\Awrc_[0-9a-f]{64}\Z")
_BROKER_WORKER_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _validated_broker_base_url(value: str) -> str:
    text = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(text)
    local_host = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme not in ({"http", "https"} if local_host else {"https"}):
        raise ValueError("warm_broker_https_required")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("warm_broker_base_url_invalid")
    if parsed.path not in {"", "/"}:
        raise ValueError("warm_broker_base_url_path_invalid")
    if not local_host:
        exact_https_origin(text)
    return text


def _validated_broker_token(value: str) -> str:
    token = value.strip()
    if len(token.encode("utf-8")) < 32:
        raise ValueError("warm_broker_token_too_short")
    return token


def _canonical_broker_job_id(value: Any) -> str:
    text = str(value or "").strip()
    if not _CANONICAL_BROKER_JOB_ID.fullmatch(text):
        raise WarmBrokerContractError("warm_broker_canonical_job_id_invalid")
    return text


def _broker_claim_token(value: Any) -> str:
    text = str(value or "").strip()
    if not _BROKER_CLAIM_TOKEN.fullmatch(text):
        raise WarmBrokerContractError("warm_broker_claim_token_invalid")
    return text


def _http_broker_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    bearer_token: str,
    extra_headers: Mapping[str, str] | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any] | None:
    data = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(dict(extra_headers))
    try:
        parsed = urllib.parse.urlsplit(url)
        allowed_origins = origins_from_env(WARM_BROKER_ALLOWED_ORIGINS_ENV)
        if parsed.scheme.lower() == "https" and not allowed_origins:
            allowed_origins = (exact_https_origin(url),)
        response = fetch_bounded_service_url(
            url,
            method=method,
            data=data,
            headers=headers,
            timeout_seconds=max(1, min(int(timeout), 600)),
            max_bytes=WARM_BROKER_MAX_RESPONSE_BYTES,
            allowed_origins=allowed_origins,
            allowed_content_types=("application/json",),
            max_redirects=0,
        )
        status_code = response.status
        raw = response.body
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise PresignedUrlAccessError(
                operation="warm_broker_request",
                status_code=exc.code,
                classification="warm_broker_unauthorized",
            ) from exc
        raise
    if status_code == 204 or not raw:
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WarmBrokerContractError("warm_broker_response_malformed") from exc
    if not isinstance(decoded, Mapping):
        raise WarmBrokerContractError("warm_broker_response_not_mapping")
    return dict(decoded)


class DurableBrokerJobSource:
    """Worker-side durable broker transport with server leases and canonical IDs."""

    def __init__(
        self,
        broker_base_url: str,
        bearer_token: str,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 900.0,
        http_request: Callable[
            [str, str, Mapping[str, Any] | None, str, Mapping[str, str] | None],
            dict[str, Any] | None,
        ] = _http_broker_json,
        max_consecutive_failures: int = 10,
    ) -> None:
        self.broker_base_url = _validated_broker_base_url(broker_base_url)
        self.bearer_token = _validated_broker_token(bearer_token)
        self.worker_id = worker_id or f"warm-worker-{uuid.uuid4().hex}"
        if not _BROKER_WORKER_ID.fullmatch(self.worker_id):
            raise ValueError("warm_broker_worker_id_invalid")
        self.lease_seconds = float(lease_seconds)
        if not 1.0 <= self.lease_seconds <= 86_400.0:
            raise ValueError("warm_broker_lease_seconds_out_of_range")
        self._http_request = http_request
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.consecutive_failures = 0
        self.last_error: str | None = None
        self._active_claim_tokens: dict[str, str] = {}

    def _record_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.last_error = reason
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise WarmInboxUnrecoverable(
                reason=reason,
                failures=self.consecutive_failures,
            )

    def poll(self) -> Optional[WarmJob]:
        try:
            payload = self._http_request(
                "POST",
                f"{self.broker_base_url}/v1/warm-render/jobs/claim",
                {"worker_id": self.worker_id, "lease_seconds": self.lease_seconds},
                self.bearer_token,
                None,
            )
        except PresignedUrlAccessError:
            raise
        except (OSError, urllib.error.HTTPError, WarmBrokerContractError):
            self._record_failure("warm_broker_claim_failed")
            return None
        if payload is None:
            self.consecutive_failures = 0
            self.last_error = None
            return None
        try:
            canonical_id = _canonical_broker_job_id(payload.get("canonical_job_id"))
            claim_token = _broker_claim_token(payload.get("claim_token"))
            scenario = payload.get("scenario")
            if not isinstance(scenario, Mapping):
                raise WarmBrokerContractError("warm_broker_scenario_not_mapping")
        except WarmBrokerContractError:
            self._record_failure("warm_broker_claim_contract_invalid")
            return None
        self.consecutive_failures = 0
        self.last_error = None
        self._active_claim_tokens[canonical_id] = claim_token
        return WarmJob(
            request_id=canonical_id,
            scenario=dict(scenario),
            stop=payload.get("stop") is True,
            session_nonce=str(payload.get("session_nonce") or ""),
            client_request_label=str(payload.get("client_request_label") or ""),
            server_idempotency_key=str(payload.get("server_idempotency_key") or ""),
        )

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        canonical_id = _canonical_broker_job_id(request_id)
        claim_token = self._active_claim_tokens.get(canonical_id)
        if claim_token is None:
            raise WarmBrokerContractError("warm_broker_active_claim_missing")
        self._http_request(
            "PUT",
            f"{self.broker_base_url}/v1/warm-render/jobs/{canonical_id}/result",
            {"claim_token": claim_token, "result": dict(result)},
            self.bearer_token,
            None,
        )
        self._active_claim_tokens.pop(canonical_id, None)


class DurableWarmPoolClient:
    """Control-plane client for the durable warm-render broker."""

    def __init__(
        self,
        broker_base_url: str,
        bearer_token: str,
        *,
        http_request: Callable[
            [str, str, Mapping[str, Any] | None, str, Mapping[str, str] | None],
            dict[str, Any] | None,
        ] = _http_broker_json,
        session_nonce: str | None = None,
    ) -> None:
        self.broker_base_url = _validated_broker_base_url(broker_base_url)
        self.bearer_token = _validated_broker_token(bearer_token)
        self._http_request = http_request
        self.session_nonce = session_nonce or uuid.uuid4().hex
        self._submission_index = 0

    def submit(
        self,
        scenario: dict[str, Any],
        request_id: Optional[str] = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        self._submission_index += 1
        client_label = str(request_id or f"job-{self._submission_index}")
        if idempotency_key is None:
            idempotency_key = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    json.dumps(
                        {
                            "client_label": client_label,
                            "scenario": scenario,
                            "session_nonce": self.session_nonce,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        response = self._http_request(
            "POST",
            f"{self.broker_base_url}/v1/warm-render/jobs",
            {
                "scenario": dict(scenario),
                "idempotency_key": idempotency_key,
                "client_request_label": client_label,
                "session_nonce": self.session_nonce,
                "stop": False,
            },
            self.bearer_token,
            None,
        )
        if response is None:
            raise WarmBrokerContractError("warm_broker_submit_response_missing")
        return _canonical_broker_job_id(response.get("canonical_job_id"))

    def submit_stop(self, *, idempotency_key: str | None = None) -> str:
        key = idempotency_key or f"stop:{self.session_nonce}"
        response = self._http_request(
            "POST",
            f"{self.broker_base_url}/v1/warm-render/jobs",
            {
                "scenario": {},
                "idempotency_key": key,
                "client_request_label": "stop",
                "session_nonce": self.session_nonce,
                "stop": True,
            },
            self.bearer_token,
            None,
        )
        if response is None:
            raise WarmBrokerContractError("warm_broker_stop_response_missing")
        return _canonical_broker_job_id(response.get("canonical_job_id"))

    def poll_result(
        self,
        request_id: str,
        *,
        timeout_s: float = 300.0,
        interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Optional[dict[str, Any]]:
        canonical_id = _canonical_broker_job_id(request_id)
        deadline = clock() + timeout_s
        while clock() < deadline:
            try:
                response = self._http_request(
                    "GET",
                    f"{self.broker_base_url}/v1/warm-render/jobs/{canonical_id}/result",
                    None,
                    self.bearer_token,
                    {"X-Warm-Session-Nonce": self.session_nonce},
                )
                if response is not None:
                    result = response.get("result")
                    if not isinstance(result, Mapping):
                        raise WarmBrokerContractError(
                            "warm_broker_result_response_invalid"
                        )
                    if _canonical_broker_job_id(
                        response.get("canonical_job_id")
                    ) != canonical_id:
                        raise WarmBrokerContractError(
                            "warm_broker_result_job_id_mismatch"
                        )
                    return dict(result)
            except urllib.error.HTTPError as exc:
                if exc.code not in {404}:
                    raise
            sleep(interval_s)
        return None


class SignedUrlJobSource:
    """Unsafe compatibility shim for the retired single-object transport.

    Public/live call sites must use :class:`DurableBrokerJobSource`. This shim is
    retained only to read historical fixtures and requires an explicit unsafe
    opt-in so a deployment cannot silently return to overwriteable inbox state.
    """

    def __init__(self, inbox_get_url: str, out_dir: Path | str, *,
                 http_get: Callable[[str], bytes] = _http_get_bytes,
                 max_consecutive_failures: int = 10,
                 allow_unsafe_single_object: bool = False) -> None:
        if not allow_unsafe_single_object:
            raise RuntimeError(
                "single_object_warm_inbox_retired_use_durable_broker"
            )
        self.inbox_get_url = inbox_get_url
        self.out_dir = Path(out_dir)
        self.results_dir = self.out_dir / "warm_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._http_get = http_get
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.consecutive_failures = 0
        self.last_error: str | None = None
        # The inbox is seeded with seq=0 at presign time; start at 0 so that seed is NOT claimed as a
        # job (the control plane's first real submit is seq=1).
        self._last_seq = 0

    def _reset_failures(self) -> None:
        self.consecutive_failures = 0
        self.last_error = None

    def _record_hard_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.last_error = reason
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise WarmInboxUnrecoverable(
                reason=reason,
                failures=self.consecutive_failures,
            )

    def poll(self) -> Optional[WarmJob]:
        try:
            raw = self._http_get(self.inbox_get_url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._reset_failures()
                return None
            reason = _http_error_classification(exc.code)
            self._record_hard_failure(reason)
            if exc.code in (401, 403):
                raise PresignedUrlAccessError(
                    operation="warm_inbox_get",
                    status_code=exc.code,
                    classification=reason,
                ) from exc
            return None
        except Exception:  # noqa: BLE001 - transient network failures are treated as no job yet
            return None
        if not raw:
            self._reset_failures()
            return None
        try:
            payload = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:  # noqa: BLE001 - repeated malformed payloads are a broken inbox
            self._record_hard_failure("warm_inbox_malformed_json")
            return None
        self._reset_failures()
        seq = int(payload.get("seq", -1))
        if seq <= self._last_seq:
            return None
        self._last_seq = seq
        return WarmJob(
            request_id=str(payload.get("request_id") or seq),
            scenario=dict(payload.get("scenario") or {}),
            stop=bool(payload.get("stop")),
            session_nonce=str(payload.get("warm_session_nonce") or ""),
        )

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        (self.results_dir / f"{request_id}.json").write_text(json.dumps(result, indent=2))


class WarmPoolClient:
    """Unsafe compatibility shim for the retired single-object client.

    Use :class:`DurableWarmPoolClient` for all live/public execution.
    """

    def __init__(self, inbox_put_url: str, output_get_url: str, *,
                 http_put: Callable[[str, bytes], None] = _http_put_bytes,
                 http_get: Callable[[str], bytes] = _http_get_bytes,
                 session_nonce: str | None = None,
                 allow_unsafe_single_object: bool = False) -> None:
        if not allow_unsafe_single_object:
            raise RuntimeError(
                "single_object_warm_inbox_retired_use_durable_broker"
            )
        self.inbox_put_url = inbox_put_url
        self.output_get_url = output_get_url
        self._http_put = http_put
        self._http_get = http_get
        self._seq = 0
        self.session_nonce = session_nonce or uuid.uuid4().hex
        self._submitted_request_ids: set[str] = set()

    def submit(self, scenario: dict[str, Any], request_id: Optional[str] = None) -> str:
        self._seq += 1
        rid = request_id or f"job-{self._seq}"
        payload = {
            "seq": self._seq,
            "request_id": rid,
            "scenario": scenario,
            "stop": False,
            "warm_session_nonce": self.session_nonce,
        }
        self._http_put(self.inbox_put_url, json.dumps(payload).encode())
        self._submitted_request_ids.add(rid)
        return rid

    def submit_stop(self) -> None:
        self._seq += 1
        self._http_put(self.inbox_put_url,
                       json.dumps({
                           "seq": self._seq,
                           "request_id": "stop",
                           "stop": True,
                           "warm_session_nonce": self.session_nonce,
                       }).encode())

    def poll_result(self, request_id: str, *, timeout_s: float = 300.0, interval_s: float = 5.0,
                    clock: Callable[[], float] = time.monotonic,
                    sleep: Callable[[float], None] = time.sleep) -> Optional[dict[str, Any]]:
        key = f"warm_results/{request_id}.json"
        deadline = clock() + timeout_s
        while clock() < deadline:
            try:
                raw = self._http_get(self.output_get_url)
                if raw:
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        if key in z.namelist():
                            result = json.loads(z.read(key).decode())
                            if result.get("warm_session_nonce") != self.session_nonce:
                                sleep(interval_s)
                                continue
                            return result
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    pass
                elif exc.code in (401, 403):
                    raise PresignedUrlAccessError(
                        operation="warm_output_get",
                        status_code=exc.code,
                        classification=_http_error_classification(exc.code),
                    ) from exc
                else:
                    pass
            except Exception:  # noqa: BLE001 - output zip not posted yet / mid-upload: retry
                pass
            sleep(interval_s)
        return None


def _safe_request_id(value: str, *, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("scenarios", [])
    if not isinstance(payload, list):
        raise ValueError("scenarios_must_be_list_or_mapping_with_scenarios")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _warm_serve_broker_files_from_manifest(
    manifest_path: str | Path,
) -> tuple[Path, Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    warm_serve = manifest.get("warm_serve") if isinstance(manifest, Mapping) else None
    if not isinstance(warm_serve, Mapping):
        raise ValueError("manifest_missing_warm_serve")
    broker_url = str(warm_serve.get("broker_base_url_file") or "").strip()
    broker_token = str(warm_serve.get("broker_token_file") or "").strip()
    if not broker_url or not broker_token:
        raise ValueError("manifest_missing_durable_warm_render_broker_files")
    return Path(broker_url), Path(broker_token)


def submit_warm_render_batch(
    *,
    manifest_path: str | Path,
    scenarios_path: str | Path,
    out_dir: str | Path,
    timeout_s: float = 900.0,
    interval_s: float = 5.0,
    stop_after: bool = False,
    session_nonce: str | None = None,
    http_put: Callable[[str, bytes], Any] = _http_put_bytes,
    http_get: Callable[[str], bytes] = _http_get_bytes,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Submit multiple scenarios through one warm-pool client session.

    Every submission is assigned a canonical server ID. The supplied scenario ID
    is only a client label and never becomes a broker key or result path.
    """
    broker_url_file, broker_token_file = _warm_serve_broker_files_from_manifest(
        manifest_path
    )
    broker_base_url = broker_url_file.read_text(encoding="utf-8").strip()
    broker_token = broker_token_file.read_text(encoding="utf-8").strip()
    scenarios = _load_scenarios(scenarios_path)
    resolved_out = Path(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)
    def legacy_http_adapter(
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        _token: str,
        _headers: Mapping[str, str] | None,
    ) -> dict[str, Any] | None:
        if method in {"POST", "PUT"}:
            response = http_put(
                url,
                json.dumps(dict(payload or {}), separators=(",", ":")).encode(),
            )
        else:
            response = http_get(url)
        if response is None or response == b"" or response == "":
            return None
        if isinstance(response, Mapping):
            return dict(response)
        raw = response.decode("utf-8") if isinstance(response, bytes) else str(response)
        decoded = json.loads(raw)
        if not isinstance(decoded, Mapping):
            raise WarmBrokerContractError("warm_broker_response_not_mapping")
        return dict(decoded)

    client = DurableWarmPoolClient(
        broker_base_url,
        broker_token,
        http_request=legacy_http_adapter,
        session_nonce=session_nonce,
    )
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            raw_id = str(
                scenario.get("scenario_id")
                or scenario.get("id")
                or scenario.get("task_id")
                or f"job-{index}"
            )
            request_id = _safe_request_id(raw_id, fallback=f"job-{index}")
            submitted = client.submit(scenario, request_id=request_id)
            result = client.poll_result(
                submitted,
                timeout_s=timeout_s,
                interval_s=interval_s,
                clock=clock,
                sleep=sleep,
            )
            record = {
                "request_id": submitted,
                "canonical_job_id": submitted,
                "client_request_label": request_id,
                "scenario_id": scenario.get("scenario_id") or scenario.get("id"),
                "result_collected": result is not None,
                "result": result,
            }
            if result is None:
                record["blocker"] = "warm_render_result_timeout"
                blockers.append("warm_render_result_timeout")
            else:
                (resolved_out / f"{submitted}.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
            records.append(record)
    except PresignedUrlAccessError as exc:
        blockers.append(exc.classification)
        records.append({
            "request_id": None,
            "result_collected": False,
            "blocker": exc.classification,
            "operation": exc.operation,
            "http_status": exc.status_code,
        })
    finally:
        if stop_after:
            try:
                client.submit_stop()
            except Exception as exc:  # noqa: BLE001
                blockers.append(f"warm_stop_submit_failed:{type(exc).__name__}")
    summary = {
        "schema_version": "warm_render_batch_client.v1",
        "status": "completed" if not blockers else "blocked",
        "scenario_count": len(scenarios),
        "results_collected": sum(1 for item in records if item.get("result_collected")),
        "blockers": sorted(set(blockers)),
        "session_nonce": client.session_nonce,
        "durable_broker_used": True,
        "single_object_transport_used": False,
        "records": records,
        "raw_url_values_recorded": False,
        "proof_boundary": (
            "Warm render batch transport only. This proves job submission/result collection "
            "from a serving pod, not Isaac render quality, WAM quality, task success, or robot readiness."
        ),
    }
    (resolved_out / "warm_render_batch_results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Submit a batch of scenarios to an Isaac warm serve pod")
    parser.add_argument("--manifest", required=True, help="isaac_g1_kitchen_parity_job_manifest.json with warm_serve URL files")
    parser.add_argument("--scenarios", required=True, help="JSON list or mapping with scenarios[] to submit")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for each warm result")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between output polls")
    parser.add_argument("--stop-after", action="store_true", help="submit a stop sentinel after the batch")
    args = parser.parse_args(argv)
    summary = submit_warm_render_batch(
        manifest_path=args.manifest,
        scenarios_path=args.scenarios,
        out_dir=args.out_dir,
        timeout_s=args.timeout,
        interval_s=args.interval,
        stop_after=args.stop_after,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") == "completed" else 1
