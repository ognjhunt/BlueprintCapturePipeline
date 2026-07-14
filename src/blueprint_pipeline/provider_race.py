"""Race a GPU launch across providers and keep the first one that actually boots.

Today's launch path fails over **sequentially**: try RunPod (warm-restart, then cold
create), and only if that is fully exhausted fall back to Vast. When a provider's pool is
*degraded* — capacity exists on paper but instances never finish booting — the sequential
path waits out the entire boot timeout on the bad provider before it even tries the next
one. The job stalls on the slowest, sickest option.

:func:`race_launch` flips that into a **race**: launch on every provider at once (one
thread each), poll each launched instance for its early boot marker, and return the FIRST
instance to show the marker — terminating every loser so nothing is left billing. A
degraded pool can no longer hold the job hostage: the healthy provider wins and the
loser's poll is cut short the instant a winner appears.

:class:`ProviderCircuitBreaker` adds memory across launches. It records per-provider
boot-success / dud outcomes over a recent window and, once a provider's recent dud-rate
*exceeds* a threshold, the racer skips it entirely (or, if every provider is tripped,
deprioritizes the sickest so the healthiest still starts first). That keeps a chronically
bad pool from being raced — and paid for — on every job.

This module is **pure orchestration**. It uses only the provider surface that already
exists — ``provider.launch(job_dir, request, *, cold=False)`` and
``provider.terminate(instance_id)`` — and never imports a provider, a cloud SDK, or a GPU
dependency. *How* a boot marker is detected is injected as ``marker_check`` (in production
it polls the signed object-store GET url for ``bootstrap.json``); the racer only cares that
it returns truthy once the instance is alive.
"""
from __future__ import annotations

import collections
import inspect
import io
import json
import math
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    mark_pending_teardown_ambiguous,
    open_pending_teardown,
    provider_state_from_inspect,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)

SCHEMA_VERSION = "provider_race.v2"


# ----------------------------- circuit breaker -----------------------------

class ProviderCircuitBreaker:
    """Per-provider boot-health memory over a sliding window of recent outcomes.

    Each launch outcome is recorded as a boot *success* or a *dud*. A provider is
    "tripped" once it has at least ``min_samples`` recent outcomes **and** its recent
    dud-rate strictly exceeds ``dud_rate_threshold``. ``min_samples`` prevents a single
    early failure from tripping a provider on no evidence; the window keeps the signal
    *recent* so a provider recovers automatically once fresh successes age the duds out.

    Thread-safe: :func:`race_launch` records outcomes from the main thread, but the lock
    keeps the breaker correct under any concurrent use.
    """

    def __init__(self, *, window: int = 10, dud_rate_threshold: float = 0.5,
                 min_samples: int = 3) -> None:
        self.window = max(1, int(window))
        self.dud_rate_threshold = float(dud_rate_threshold)
        self.min_samples = max(1, int(min_samples))
        # name -> deque[bool]  (True == dud, False == boot success), capped at `window`
        self._recent: dict[str, "collections.deque[bool]"] = {}
        # name -> lifetime {"success": int, "dud": int} for snapshots/telemetry
        self._totals: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    # -- recording -------------------------------------------------------

    def _bucket(self, name: str) -> "collections.deque[bool]":
        dq = self._recent.get(name)
        if dq is None:
            dq = collections.deque(maxlen=self.window)
            self._recent[name] = dq
            self._totals[name] = {"success": 0, "dud": 0}
        return dq

    def record_success(self, name: str) -> None:
        with self._lock:
            self._bucket(name).append(False)
            self._totals[name]["success"] += 1

    def record_dud(self, name: str) -> None:
        with self._lock:
            self._bucket(name).append(True)
            self._totals[name]["dud"] += 1

    # -- queries ---------------------------------------------------------

    def _dud_rate_locked(self, name: str) -> float:
        dq = self._recent.get(name)
        if not dq:
            return 0.0
        return sum(1 for is_dud in dq if is_dud) / len(dq)

    def _tripped_locked(self, name: str) -> bool:
        dq = self._recent.get(name)
        if not dq or len(dq) < self.min_samples:
            return False
        return self._dud_rate_locked(name) > self.dud_rate_threshold

    def dud_rate(self, name: str) -> float:
        """Fraction of recent outcomes that were duds (0.0 if no samples yet)."""
        with self._lock:
            return self._dud_rate_locked(name)

    def samples(self, name: str) -> int:
        with self._lock:
            return len(self._recent.get(name, ()))

    def is_tripped(self, name: str) -> bool:
        """True once recent dud-rate EXCEEDS the threshold with enough samples to trust."""
        with self._lock:
            return self._tripped_locked(name)

    # a tripped provider is one the racer should skip — same predicate, intent-named alias
    def should_skip(self, name: str) -> bool:
        return self.is_tripped(name)

    # -- provider-list helpers (skip / deprioritize) ---------------------

    def partition(self, providers: Sequence) -> "tuple[list, list]":
        """Split providers into ``(runnable, skipped)`` — skipped are the tripped ones."""
        runnable, skipped = [], []
        for provider in providers:
            (skipped if self.is_tripped(provider.name) else runnable).append(provider)
        return runnable, skipped

    def order(self, providers: Sequence) -> list:
        """Return providers healthiest-first (ascending recent dud-rate, name as tiebreak)."""
        return sorted(providers, key=lambda p: (self.dud_rate(p.name), p.name))

    def snapshot(self) -> dict:
        """Per-provider lifetime counts + recent health, for manifests / debugging."""
        with self._lock:
            out: dict[str, dict] = {}
            for name, dq in self._recent.items():
                totals = self._totals[name]
                out[name] = {
                    "success": totals["success"],
                    "dud": totals["dud"],
                    "recent_samples": len(dq),
                    "recent_dud_rate": round(self._dud_rate_locked(name), 3),
                    "tripped": self._tripped_locked(name),
                }
            return out


# ----------------------------- race launcher -----------------------------

def _safe_segment(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name)) or "provider"


def _resolve_request(request, provider, job_dir: Path | None = None):
    """``request`` may be one body shared by all providers, or a callable that builds the
    provider-native body per provider (RunPod pod body vs Vast offer-search differ)."""
    if not callable(request):
        return request
    if job_dir is not None:
        try:
            return request(provider, job_dir)
        except TypeError:
            pass
    return request(provider)


def _resolve_launch_kwargs(launch_kwargs, provider) -> dict[str, Any]:
    """Optional provider-specific kwargs forwarded to ``provider.launch``.

    ``launch_kwargs`` mirrors ``request``: either one mapping shared by every provider, or a callable
    ``launch_kwargs(provider) -> mapping``. The race keeps this optional so legacy tests/providers with
    only ``launch(job_dir, request, *, cold=False)`` continue to work.
    """
    if launch_kwargs is None:
        return {}
    value = launch_kwargs(provider) if callable(launch_kwargs) else launch_kwargs
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("launch_kwargs must be a mapping or callable returning a mapping")
    return dict(value)


def _supported_launch_kwargs(
    provider: object, requested: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Filter signature-incompatible kwargs before the sole mutation call."""
    signature = inspect.signature(provider.launch)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(requested), []
    supported = set(signature.parameters)
    accepted = {key: value for key, value in requested.items() if key in supported}
    dropped = sorted(set(requested) - set(accepted))
    return accepted, dropped


def _resolve_prelaunch_guard(prelaunch_guard, provider) -> dict[str, Any]:
    """Resolve the optional fail-closed spend guard for one provider."""
    if prelaunch_guard is None:
        return {}
    value = prelaunch_guard(provider) if callable(prelaunch_guard) else prelaunch_guard
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("prelaunch_guard must be a mapping or callable returning a mapping")
    return dict(value)


def _guard_blockers(guard: Mapping[str, Any]) -> list[str]:
    """Return blockers when a supplied prelaunch guard does not authorize launch."""
    if not guard:
        return []
    if guard.get("can_launch") is True:
        return []
    raw_blockers = guard.get("blockers")
    if isinstance(raw_blockers, str):
        blockers = [raw_blockers]
    elif isinstance(raw_blockers, Sequence) and not isinstance(raw_blockers, (bytes, bytearray)):
        blockers = [str(item) for item in raw_blockers if str(item or "").strip()]
    else:
        blockers = []
    return ["prelaunch_spend_guard_not_passed", *blockers]


def _pending_teardown_run_id(provider_name: str, idx: int) -> str:
    return f"race-{idx}-{_safe_segment(provider_name)}-{time.time_ns()}"


def _marker_check_before_deadline(
    marker_check: Callable[[object, dict], bool],
    provider: object,
    launch: dict,
    remaining_seconds: float,
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[bool, bool]:
    """Run a read-only marker probe without letting it exceed the paid deadline."""
    holder: dict[str, Any] = {}

    def _probe() -> None:
        try:
            holder["value"] = bool(marker_check(provider, launch))
        except Exception as exc:  # noqa: BLE001 - caller records probe failure
            holder["error"] = exc

    thread = threading.Thread(target=_probe, name="bounded-marker-probe", daemon=True)
    thread.start()
    timeout = max(0.0, float(remaining_seconds))
    if cancel_event is None:
        thread.join(timeout=timeout)
    else:
        deadline = time.monotonic() + timeout
        while thread.is_alive():
            if cancel_event.is_set():
                return False, False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=min(0.05, remaining))
    if thread.is_alive():
        return False, True
    if "error" in holder:
        raise holder["error"]
    return bool(holder.get("value")), False


def _teardown_proof_from_provider_action(
    provider: object,
    instance_id: str,
    teardown: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    provider_name = str(getattr(provider, "name", "") or "unknown").strip()
    action_text = str(action or "").strip().lower()
    status = str(teardown.get("status") or "").strip().lower()
    if action_text == "stop":
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=status or "stopped_for_warm_reuse",
        )
    verification: dict[str, Any] = {}
    if hasattr(provider, "inspect"):
        try:
            verification = provider_state_from_inspect(provider.inspect(instance_id))
        except Exception as exc:  # noqa: BLE001 - failed verification is evidence
            verification = {
                "api_confirmed": False,
                "provider_status": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    observed_status = str(verification.get("provider_status") or "").strip().lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=provider_name,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=utc_now_iso() if status == "terminated" else None,
    )


def boot_marker_present(
    job_dir: str | Path | None = None,
    *,
    get_url: str | None = None,
    output_url_file: str | Path | None = None,
    marker_name: str = "bootstrap.json",
    expected_launch_session_id: str | None = None,
    timeout: float = 60.0,
    urlopen: Callable[..., object] | None = None,
) -> bool:
    """Return whether a provider output zip contains the expected boot marker.

    The helper is intentionally fail-closed: missing signed URLs, expired/forbidden URLs,
    malformed zips, absent marker files, invalid marker JSON, and stale launch-session
    markers all return ``False``. It never logs or returns the signed URL.
    """
    try:
        resolved_url = str(get_url or "").strip()
        if not resolved_url:
            url_file = Path(output_url_file) if output_url_file else (
                Path(job_dir) / "provider_output_get_url.txt" if job_dir is not None else None
            )
            if url_file is None or not url_file.is_file():
                return False
            resolved_url = url_file.read_text(encoding="utf-8").strip()
        if not resolved_url:
            return False

        opener = urlopen or urllib.request.urlopen
        try:
            response = opener(resolved_url, timeout=timeout)
        except TypeError:
            response = opener(resolved_url)
        data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if marker_name not in zf.namelist():
                return False
            marker = json.loads(zf.read(marker_name).decode())
        if not isinstance(marker, dict):
            return False
        expected = str(expected_launch_session_id or "").strip()
        if expected and str(marker.get("launch_session_id") or "") != expected:
            return False
        return True
    except Exception:  # noqa: BLE001 - boot-marker probes must fail closed, not crash launch races
        return False


def race_launch(
    providers: Sequence,
    request,
    marker_check: Callable[[object, dict], bool],
    marker_timeout: float,
    *,
    job_dir,
    cold: bool = False,
    poll_interval: float = 10.0,
    circuit_breaker: ProviderCircuitBreaker | None = None,
    terminate_losers: bool = True,
    launch_kwargs: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    bundle_kind: str | None = None,
    readiness_marker: str | None = None,
    prelaunch_guard: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    pending_teardown_lane: str | None = None,
    pending_teardown_max_age_seconds: int = 7200,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Run a provider race with known-allocation cleanup on every BaseException."""
    interrupt_state: dict[str, Any] = {}
    try:
        return _race_launch_impl(
            providers,
            request,
            marker_check,
            marker_timeout,
            job_dir=job_dir,
            cold=cold,
            poll_interval=poll_interval,
            circuit_breaker=circuit_breaker,
            terminate_losers=terminate_losers,
            launch_kwargs=launch_kwargs,
            bundle_kind=bundle_kind,
            readiness_marker=readiness_marker,
            prelaunch_guard=prelaunch_guard,
            pending_teardown_lane=pending_teardown_lane,
            pending_teardown_max_age_seconds=pending_teardown_max_age_seconds,
            sleep=sleep,
            monotonic=monotonic,
            _interrupt_state=interrupt_state,
        )
    except BaseException:
        cleanup = interrupt_state.get("cleanup")
        if callable(cleanup):
            cleanup()
        raise


def _race_launch_impl(
    providers: Sequence,
    request,
    marker_check: Callable[[object, dict], bool],
    marker_timeout: float,
    *,
    job_dir,
    cold: bool = False,
    poll_interval: float = 10.0,
    circuit_breaker: ProviderCircuitBreaker | None = None,
    terminate_losers: bool = True,
    launch_kwargs: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    bundle_kind: str | None = None,
    readiness_marker: str | None = None,
    prelaunch_guard: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    pending_teardown_lane: str | None = None,
    pending_teardown_max_age_seconds: int = 7200,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    _interrupt_state: dict[str, Any] | None = None,
) -> dict:
    """Launch on every provider at once; return the first instance to show its boot marker.

    Args:
        providers: provider objects exposing ``launch(job_dir, request, *, cold=False)``
            and ``terminate(instance_id)``. Nothing else is touched.
        request: the launch body. Either a single value passed to every provider, or a
            callable ``request(provider) -> body`` to build a provider-native body each.
        marker_check: ``marker_check(provider, launch_result) -> bool`` — polled per
            launched instance until truthy (booted) or the timeout elapses. This is the
            injected seam that, in production, polls the signed object-store url for the
            early ``bootstrap.json`` marker.
        marker_timeout: seconds to wait for a launched instance's boot marker before
            declaring it a dud. Polling is discretized into
            ``ceil(marker_timeout / poll_interval)`` attempts (mirrors ``watch_and_collect``).
        job_dir: working dir; each contender gets an isolated subdir so concurrent
            ``launch`` calls can't clobber one another's ``started_*_id.txt``.
        cold: forwarded verbatim to every ``provider.launch(..., cold=cold)``.
        poll_interval: seconds between boot-marker polls.
        circuit_breaker: optional :class:`ProviderCircuitBreaker`. Tripped providers are
            skipped (unless *all* are tripped, in which case they race healthiest-first so
            the job isn't dead-ended). Outcomes are recorded back into it.
        terminate_losers: when True (default), every launched non-winner is terminated.
        launch_kwargs: optional mapping, or callable returning a mapping, forwarded to
            ``provider.launch``. This lets the job enforce warm-only semantics with
            ``allow_cold_fallback=False`` without baking provider-specific options into the racer.
        prelaunch_guard: optional mapping, or callable returning a mapping, that must
            carry ``can_launch: true`` before the provider's ``launch`` method is
            called. A failed guard records a contender-level
            ``prelaunch_blocked`` outcome and makes zero provider API calls.
        pending_teardown_lane: optional paid-lane name. When supplied, every contender
            that passes its prelaunch guard opens a ``pending_teardown.v1`` record
            immediately before ``provider.launch``. The winning record is returned to
            the caller; loser records are closed only with provider-API teardown proof.
        pending_teardown_max_age_seconds: orphan-reaper max age for records opened by
            ``pending_teardown_lane``.
        sleep / monotonic: injectable clocks for hermetic, fast tests.

    Returns:
        A dict with ``status`` ("launched"|"blocked"), the winning ``provider`` name /
        ``instance_id`` / ``mode``, the live ``winner_provider`` object (for the caller's
        watch+collect), the per-provider ``contenders`` records, the ``skipped`` provider
        names, and (when blocked) a ``reason``.
    """
    providers = list(providers)
    job_root = Path(job_dir)

    # 1) Consult the breaker: skip tripped providers; if all are tripped, race them anyway
    #    (healthiest first) rather than dead-end the job.
    skipped_names: list[str] = []
    if circuit_breaker is not None and providers:
        runnable, skipped = circuit_breaker.partition(providers)
        if runnable:
            runnable = circuit_breaker.order(runnable)
            skipped_names = [p.name for p in skipped]
        else:
            runnable = circuit_breaker.order(providers)  # everyone tripped -> race all
    else:
        runnable = providers

    if not runnable:
        return _result(
            None,
            None,
            [],
            skipped_names,
            0,
            reason="no_providers",
            bundle_kind=bundle_kind,
            readiness_marker=readiness_marker,
        )

    # 2) Shared race state. `winner_found` lets a losing thread abort its poll the instant
    #    someone wins, so a degraded provider never makes the job wait out its full timeout.
    winner_lock = threading.Lock()
    winner_found = threading.Event()
    state = {"winner": None}  # index of the winning contender, set once under the lock
    records = [
        {"provider": p.name, "index": i, "outcome": None, "instance_id": None,
         "mode": None, "launch": None, "terminated": None, "reason": None,
         "stopped": None, "teardown_action": None, "launch_kwargs": None,
         "prelaunch_guard": None, "pending_teardown_record": None,
         "pending_teardown_status": None, "blockers": [],
         "polls": 0, "elapsed_seconds": 0.0}
        for i, p in enumerate(runnable)
    ]
    interrupt_cleanup_requested = threading.Event()
    started_threads: list[threading.Thread] = []
    attempts = 1 if poll_interval <= 0 else max(1, math.ceil(marker_timeout / poll_interval))

    def _cleanup_interrupted_contender(idx: int, provider: object) -> None:
        """Terminate a known id owned by a race that is no longer returning a winner."""
        rec = records[idx]
        # A contender thread and the coordinator can both enter interrupt
        # cleanup. Claim the teardown once under the shared race lock before
        # making the provider call, otherwise the same paid instance can be
        # terminated twice depending on thread scheduling.
        with winner_lock:
            iid = rec.get("instance_id")
            if not iid or rec.get("interruption_cleanup_attempted"):
                return
            rec["interruption_cleanup_attempted"] = True
        rec["teardown_action"] = "terminate"
        try:
            rec["terminated"] = provider.terminate(iid)
        except Exception as exc:  # noqa: BLE001 - proof remains fail-closed
            rec["terminated"] = {
                "status": "terminate_failed",
                "error_type": type(exc).__name__,
            }
        if not rec.get("pending_teardown_record"):
            return
        try:
            proof = _teardown_proof_from_provider_action(
                provider,
                str(iid),
                rec.get("terminated")
                if isinstance(rec.get("terminated"), Mapping)
                else {},
                "terminate",
            )
            rec["teardown_proof"] = proof
        except Exception as exc:  # noqa: BLE001 - preserve the open obligation
            rec["pending_teardown_status"] = "proof_failed"
            rec["teardown_proof_error_type"] = type(exc).__name__
            return
        try:
            closure = close_pending_teardown(rec["pending_teardown_record"], proof)
            rec["pending_teardown_status"] = closure.get("status")
        except Exception as exc:  # noqa: BLE001 - preserve the open obligation
            rec["pending_teardown_status"] = "close_failed"
            rec["pending_teardown_close_error_type"] = type(exc).__name__

    def _cleanup_interrupted_race() -> None:
        interrupt_cleanup_requested.set()
        winner_found.set()
        for thread in tuple(started_threads):
            if thread.ident is None:
                continue
            while thread.is_alive():
                try:
                    thread.join(timeout=0.1)
                except BaseException:
                    # Keep the original interrupt pending until all known-id
                    # cleanup has had a chance to run.
                    continue
        for idx, provider in enumerate(runnable):
            _cleanup_interrupted_contender(idx, provider)

    if _interrupt_state is not None:
        _interrupt_state["cleanup"] = _cleanup_interrupted_race

    def _run(idx: int, provider) -> None:
        rec = records[idx]
        started = monotonic()
        sub_dir = job_root / f"contender-{idx}-{_safe_segment(provider.name)}"
        try:
            sub_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 — never let dir setup sink the contender
            pass

        # -- launch --
        try:
            guard = _resolve_prelaunch_guard(prelaunch_guard, provider)
            rec["prelaunch_guard"] = guard or None
            blockers = _guard_blockers(guard)
            if blockers:
                rec["outcome"] = "prelaunch_blocked"
                rec["reason"] = blockers[0]
                rec["blockers"] = blockers
                rec["elapsed_seconds"] = round(monotonic() - started, 3)
                return
            requested_kwargs = _resolve_launch_kwargs(launch_kwargs, provider)
            kwargs, unsupported_kwargs = _supported_launch_kwargs(
                provider, requested_kwargs
            )
            request_body = _resolve_request(request, provider, sub_dir)
            pending_record: dict[str, Any] | None = None
            if pending_teardown_lane:
                pending_record = open_pending_teardown(
                    provider=provider.name,
                    lane=str(pending_teardown_lane),
                    run_id=_pending_teardown_run_id(provider.name, idx),
                    job_dir=sub_dir,
                    max_age_seconds=max(1, int(pending_teardown_max_age_seconds)),
                )
                rec["pending_teardown_record"] = pending_record["path"]
                if isinstance(request_body, Mapping):
                    request_body = dict(request_body)
                    request_body["pending_teardown_record"] = pending_record["path"]
            rec["launch_kwargs_requested"] = dict(requested_kwargs)
            rec["launch_kwargs"] = dict(kwargs)
            rec["launch_kwargs_unsupported"] = unsupported_kwargs
            launch = provider.launch(
                sub_dir,
                request_body,
                cold=cold,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — a thrown launch is just a dud, not a crash
            rec["outcome"] = "launch_ambiguous"
            rec["reason"] = ("launch_raised:" + repr(exc))[:200]
            if rec.get("pending_teardown_record"):
                try:
                    mark_pending_teardown_ambiguous(
                        rec["pending_teardown_record"],
                        reason="provider_launch_raised_before_allocation",
                        evidence={"error_type": type(exc).__name__},
                    )
                except Exception as mark_exc:  # noqa: BLE001 - in-memory state wins
                    rec["pending_teardown_mark_error_type"] = type(mark_exc).__name__
                rec["pending_teardown_status"] = "open_ambiguous_allocation"
            rec["elapsed_seconds"] = round(monotonic() - started, 3)
            return
        if isinstance(launch, dict):
            launch.setdefault("job_dir", str(sub_dir))
            if rec.get("pending_teardown_record"):
                launch["pending_teardown_record"] = rec["pending_teardown_record"]
        rec["launch"] = launch if isinstance(launch, dict) else {"raw": repr(launch)[:200]}
        iid = launch.get("instance_id") if isinstance(launch, dict) else None
        if iid:
            rec["instance_id"] = str(iid)
            rec["mode"] = launch.get("mode") if isinstance(launch, Mapping) else None
        if iid and rec.get("pending_teardown_record"):
            try:
                bind_pending_teardown_instance(rec["pending_teardown_record"], str(iid))
            except Exception as exc:  # noqa: BLE001 - known id remains cleanup-owned
                rec["pending_teardown_status"] = "bind_failed"
                rec["pending_teardown_bind_error_type"] = type(exc).__name__
                rec["outcome"] = "partial_allocation"
        if interrupt_cleanup_requested.is_set() and iid:
            rec["outcome"] = "interrupted"
            rec["reason"] = "provider_race_interrupted"
            rec["elapsed_seconds"] = round(monotonic() - started, 3)
            _cleanup_interrupted_contender(idx, provider)
            return
        launched = bool(
            isinstance(launch, dict)
            and launch.get("status") == "launched"
            and iid
            and rec.get("outcome") != "partial_allocation"
        )
        if not launched:
            if iid:
                rec["outcome"] = "partial_allocation"
                rec["reason"] = "nonlaunched_response_with_instance_id"
            if rec.get("pending_teardown_record") and not iid:
                if isinstance(launch, Mapping) and launch.get("allocation_created") is False:
                    try:
                        cancel_pending_teardown(
                            rec["pending_teardown_record"],
                            reason="launch_returned_explicit_no_allocation",
                            evidence=launch,
                        )
                        rec["pending_teardown_status"] = "cancelled_no_allocation"
                    except Exception as exc:  # noqa: BLE001 - block safe retry
                        rec["pending_teardown_status"] = "cancel_failed"
                        rec["pending_teardown_cancel_error_type"] = type(exc).__name__
                else:
                    rec["pending_teardown_status"] = "open_ambiguous_allocation"
                    rec["outcome"] = "launch_ambiguous"
                    try:
                        mark_pending_teardown_ambiguous(
                            rec["pending_teardown_record"],
                            reason="launch_returned_without_explicit_no_allocation",
                            evidence={
                                "status": launch.get("status") if isinstance(launch, Mapping) else None,
                                "blockers": launch.get("blockers") if isinstance(launch, Mapping) else None,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 - in-memory state wins
                        rec["pending_teardown_mark_error_type"] = type(exc).__name__
            if rec.get("outcome") not in {"launch_ambiguous", "partial_allocation"}:
                rec["outcome"] = "no_capacity"
            blockers = launch.get("blockers") if isinstance(launch, dict) else None
            rec["reason"] = (blockers[0] if blockers else "launch_not_launched")
            rec["elapsed_seconds"] = round(monotonic() - started, 3)
            return
        # -- poll for the early boot marker --
        booted = aborted = False
        # Readiness timeout starts only after the provider returns a known
        # allocation. Slow create calls and fail-closed teardown-record I/O must
        # not consume the worker's actual boot-marker observation window.
        deadline = monotonic() + max(0.0, float(marker_timeout))
        for attempt in range(attempts):
            rec["polls"] = attempt + 1
            if interrupt_cleanup_requested.is_set():
                aborted = True
                rec["reason"] = "provider_race_interrupted"
                break
            if winner_found.is_set():
                aborted = True  # someone else already won -> stop waiting on this instance
                break
            remaining = deadline - monotonic()
            if remaining <= 0:
                rec["reason"] = "marker_wall_clock_deadline"
                break
            try:
                marker_value, probe_timed_out = _marker_check_before_deadline(
                    marker_check,
                    provider,
                    launch,
                    remaining,
                    cancel_event=interrupt_cleanup_requested,
                )
                if interrupt_cleanup_requested.is_set():
                    aborted = True
                    rec["reason"] = "provider_race_interrupted"
                    break
                if probe_timed_out:
                    rec["reason"] = "marker_probe_deadline_exhausted"
                    break
                if marker_value:
                    booted = True
                    break
            except Exception as exc:  # noqa: BLE001 — a flaky probe is not a boot
                rec["reason"] = ("marker_check_raised:" + repr(exc))[:200]
            if attempt < attempts - 1:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                sleep(min(float(poll_interval), remaining))
        rec["elapsed_seconds"] = round(monotonic() - started, 3)

        # -- classify --
        if booted:
            with winner_lock:
                if state["winner"] is None:
                    state["winner"] = idx
                    rec["outcome"] = "won"
                    winner_found.set()
                    return
            rec["outcome"] = "booted_lost"  # booted, but another instance got there first
            rec["reason"] = rec["reason"] or "won_but_not_first"
        elif aborted:
            rec["outcome"] = "aborted"      # cut short by the winner; not the pool's fault
            rec["reason"] = rec["reason"] or "winner_found_elsewhere"
        else:
            rec["outcome"] = "no_boot"      # launched but never showed the marker in time
            rec["reason"] = rec["reason"] or "marker_timeout"
        if interrupt_cleanup_requested.is_set():
            _cleanup_interrupted_contender(idx, provider)

    threads = [
        threading.Thread(target=_run, args=(i, p), name=f"race-{p.name}-{i}", daemon=True)
        for i, p in enumerate(runnable)
    ]
    for thread in threads:
        # Register before start so an exception from Thread.start itself cannot
        # create an untracked contender.
        started_threads.append(thread)
        thread.start()
    for thread in started_threads:
        thread.join()  # every launch has returned -> every instance_id is known, nothing leaks

    for rec in records:
        if rec.get("outcome") is None and rec.get("pending_teardown_record"):
            rec["outcome"] = "launch_ambiguous"
            rec["reason"] = "contender_ended_without_safe_outcome"
            rec["pending_teardown_status"] = "open_ambiguous_allocation"

    winner_idx = state["winner"]
    ambiguous_launch = any(
        rec.get("outcome") == "launch_ambiguous" for rec in records
    )
    partial_allocation = any(
        rec.get("outcome") == "partial_allocation" for rec in records
    )
    if ambiguous_launch or partial_allocation:
        # A lost create response may have allocated an unknown instance. Do not
        # promote a known winner while that competing mutation is unresolved.
        winner_idx = None

    # 3) Tear down every launched loser (winner is kept for the caller to watch+collect).
    terminated = 0
    for i, rec in enumerate(records):
        if i == winner_idx:
            continue
        iid = rec["instance_id"]
        if iid and terminate_losers:
            try:
                mode = str(rec.get("mode") or "")
                if (
                    mode.startswith("warm")
                    and hasattr(runnable[i], "stop")
                    and not pending_teardown_lane
                ):
                    rec["teardown_action"] = "stop"
                    rec["stopped"] = runnable[i].stop(iid)
                else:
                    rec["teardown_action"] = "terminate"
                    rec["terminated"] = runnable[i].terminate(iid)
            except Exception as exc:  # noqa: BLE001
                action = rec.get("teardown_action") or "teardown"
                if action == "stop":
                    rec["stopped"] = {"status": "stop_failed", "error": repr(exc)[:200]}
                else:
                    rec["terminated"] = {"status": "terminate_failed", "error": repr(exc)[:200]}
            if rec.get("pending_teardown_record"):
                teardown = rec.get("stopped") if rec.get("teardown_action") == "stop" else rec.get("terminated")
                proof = _teardown_proof_from_provider_action(
                    runnable[i],
                    str(iid),
                    teardown if isinstance(teardown, Mapping) else {},
                    str(rec.get("teardown_action") or "terminate"),
                )
                rec["teardown_proof"] = proof
                try:
                    closure = close_pending_teardown(
                        rec["pending_teardown_record"], proof
                    )
                    rec["pending_teardown_status"] = closure.get("status")
                except Exception as exc:  # noqa: BLE001 - continue all cleanup
                    rec["pending_teardown_status"] = "close_failed"
                    rec["pending_teardown_close_error_type"] = type(exc).__name__
            terminated += 1

    loser_cleanup_failed = bool(
        pending_teardown_lane
        and any(
            rec.get("pending_teardown_record")
            and rec.get("pending_teardown_status")
            not in {"closed", "cancelled_no_allocation"}
            for i, rec in enumerate(records)
            if i != winner_idx
        )
    )
    if loser_cleanup_failed and winner_idx is not None:
        # Never promote a winner while another paid contender may still bill.
        winner_rec = records[winner_idx]
        winner_iid = winner_rec.get("instance_id")
        if winner_iid:
            try:
                winner_rec["teardown_action"] = "terminate"
                winner_rec["terminated"] = runnable[winner_idx].terminate(winner_iid)
            except Exception as exc:  # noqa: BLE001 - proof below remains fail-closed
                winner_rec["terminated"] = {
                    "status": "terminate_failed",
                    "error_type": type(exc).__name__,
                }
            if winner_rec.get("pending_teardown_record"):
                proof = _teardown_proof_from_provider_action(
                    runnable[winner_idx],
                    str(winner_iid),
                    winner_rec.get("terminated") or {},
                    "terminate",
                )
                winner_rec["teardown_proof"] = proof
                try:
                    closure = close_pending_teardown(
                        winner_rec["pending_teardown_record"], proof
                    )
                    winner_rec["pending_teardown_status"] = closure.get("status")
                except Exception as exc:  # noqa: BLE001 - cleanup remains blocked
                    winner_rec["pending_teardown_status"] = "close_failed"
                    winner_rec["pending_teardown_close_error_type"] = (
                        type(exc).__name__
                    )
            winner_rec["outcome"] = "winner_terminated_due_unverified_loser"
            terminated += 1
        winner_idx = None

    # 4) Feed outcomes back into the breaker. A provider that booted (won OR booted_lost)
    #    is a success; one that couldn't launch or never booted is a dud. A contender merely
    #    cut short by the winner ("aborted") is neutral — it might have been perfectly healthy.
    if circuit_breaker is not None:
        for rec in records:
            outcome = rec["outcome"]
            if outcome in ("won", "booted_lost"):
                circuit_breaker.record_success(rec["provider"])
            elif outcome in ("no_boot", "no_capacity"):
                circuit_breaker.record_dud(rec["provider"])

    if winner_idx is None:
        blocked_reason = (
            "provider_launch_outcome_ambiguous"
            if ambiguous_launch
            else "provider_launch_returned_nonlaunched_allocation"
            if partial_allocation
            else "provider_race_teardown_unverified"
            if loser_cleanup_failed
            else "prelaunch_spend_guard_not_passed"
            if records and all(rec.get("outcome") == "prelaunch_blocked" for rec in records)
            else "all_providers_dudded"
        )
        return _result(
            None,
            None,
            records,
            skipped_names,
            terminated,
            reason=blocked_reason,
            bundle_kind=bundle_kind,
            readiness_marker=readiness_marker,
        )
    return _result(runnable[winner_idx], records[winner_idx], records, skipped_names,
                   terminated, reason=None, bundle_kind=bundle_kind,
                   readiness_marker=readiness_marker)


def _result(
    winner_provider,
    win_rec,
    records,
    skipped_names,
    terminated,
    *,
    reason,
    bundle_kind: str | None = None,
    readiness_marker: str | None = None,
) -> dict:
    """Assemble the uniform race result (also the no-providers / all-dudded blocked shape)."""
    paid_records = [
        record for record in records if record.get("pending_teardown_record")
    ]
    paid_retry_safe = (
        all(
            record.get("pending_teardown_status")
            in {"closed", "cancelled_no_allocation"}
            for record in paid_records
        )
        if paid_records
        else None
    )
    if win_rec is not None:
        return {
            "schema": SCHEMA_VERSION,
            "status": "launched",
            "provider": win_rec["provider"],
            "instance_id": win_rec["instance_id"],
            "mode": win_rec["mode"],
            "winner_provider": winner_provider,
            "winner_launch": win_rec["launch"],
            "pending_teardown_record": win_rec.get("pending_teardown_record"),
            "bundle_kind": bundle_kind,
            "readiness_marker": readiness_marker,
            "contenders": records,
            "skipped": skipped_names,
            "terminated_losers": terminated,
            "reason": None,
            "paid_retry_safe": paid_retry_safe,
        }
    return {
        "schema": SCHEMA_VERSION,
        "status": "blocked",
        "provider": None,
        "instance_id": None,
        "mode": None,
        "winner_provider": None,
        "winner_launch": None,
        "pending_teardown_record": None,
        "bundle_kind": bundle_kind,
        "readiness_marker": readiness_marker,
        "contenders": records,
        "skipped": skipped_names,
        "terminated_losers": terminated,
        "reason": reason,
        "paid_retry_safe": paid_retry_safe,
    }
