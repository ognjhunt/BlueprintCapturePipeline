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
import io
import json
import math
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "provider_race.v1"


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
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
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
        return _result(None, None, [], skipped_names, 0, reason="no_providers")

    # 2) Shared race state. `winner_found` lets a losing thread abort its poll the instant
    #    someone wins, so a degraded provider never makes the job wait out its full timeout.
    winner_lock = threading.Lock()
    winner_found = threading.Event()
    state = {"winner": None}  # index of the winning contender, set once under the lock
    records = [
        {"provider": p.name, "index": i, "outcome": None, "instance_id": None,
         "mode": None, "launch": None, "terminated": None, "reason": None,
         "stopped": None, "teardown_action": None, "launch_kwargs": None,
         "polls": 0, "elapsed_seconds": 0.0}
        for i, p in enumerate(runnable)
    ]
    attempts = 1 if poll_interval <= 0 else max(1, math.ceil(marker_timeout / poll_interval))

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
            kwargs = _resolve_launch_kwargs(launch_kwargs, provider)
            request_body = _resolve_request(request, provider, sub_dir)
            rec["launch_kwargs"] = dict(kwargs)
            try:
                launch = provider.launch(
                    sub_dir,
                    request_body,
                    cold=cold,
                    **kwargs,
                )
            except TypeError as exc:
                if not kwargs:
                    raise
                rec["launch_kwargs_legacy_fallback"] = repr(exc)[:200]
                launch = provider.launch(
                    sub_dir,
                    request_body,
                    cold=cold,
                )
        except Exception as exc:  # noqa: BLE001 — a thrown launch is just a dud, not a crash
            rec["outcome"] = "no_capacity"
            rec["reason"] = ("launch_raised:" + repr(exc))[:200]
            rec["elapsed_seconds"] = round(monotonic() - started, 3)
            return
        if isinstance(launch, dict):
            launch.setdefault("job_dir", str(sub_dir))
        rec["launch"] = launch if isinstance(launch, dict) else {"raw": repr(launch)[:200]}
        iid = launch.get("instance_id") if isinstance(launch, dict) else None
        launched = isinstance(launch, dict) and launch.get("status") == "launched" and bool(iid)
        if not launched:
            rec["outcome"] = "no_capacity"
            blockers = launch.get("blockers") if isinstance(launch, dict) else None
            rec["reason"] = (blockers[0] if blockers else "launch_not_launched")
            rec["elapsed_seconds"] = round(monotonic() - started, 3)
            return
        rec["instance_id"] = iid
        rec["mode"] = launch.get("mode")

        # -- poll for the early boot marker --
        booted = aborted = False
        for attempt in range(attempts):
            rec["polls"] = attempt + 1
            if winner_found.is_set():
                aborted = True  # someone else already won -> stop waiting on this instance
                break
            try:
                if marker_check(provider, launch):
                    booted = True
                    break
            except Exception as exc:  # noqa: BLE001 — a flaky probe is not a boot
                rec["reason"] = ("marker_check_raised:" + repr(exc))[:200]
            if attempt < attempts - 1:
                sleep(poll_interval)
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

    threads = [
        threading.Thread(target=_run, args=(i, p), name=f"race-{p.name}-{i}", daemon=True)
        for i, p in enumerate(runnable)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()  # every launch has returned -> every instance_id is known, nothing leaks

    winner_idx = state["winner"]

    # 3) Tear down every launched loser (winner is kept for the caller to watch+collect).
    terminated = 0
    for i, rec in enumerate(records):
        if i == winner_idx:
            continue
        iid = rec["instance_id"]
        if iid and terminate_losers:
            try:
                mode = str(rec.get("mode") or "")
                if mode.startswith("warm") and hasattr(runnable[i], "stop"):
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
            terminated += 1

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
        return _result(None, None, records, skipped_names, terminated,
                       reason="all_providers_dudded")
    return _result(runnable[winner_idx], records[winner_idx], records, skipped_names,
                   terminated, reason=None)


def _result(winner_provider, win_rec, records, skipped_names, terminated, *, reason) -> dict:
    """Assemble the uniform race result (also the no-providers / all-dudded blocked shape)."""
    if win_rec is not None:
        return {
            "schema": SCHEMA_VERSION,
            "status": "launched",
            "provider": win_rec["provider"],
            "instance_id": win_rec["instance_id"],
            "mode": win_rec["mode"],
            "winner_provider": winner_provider,
            "winner_launch": win_rec["launch"],
            "contenders": records,
            "skipped": skipped_names,
            "terminated_losers": terminated,
            "reason": None,
        }
    return {
        "schema": SCHEMA_VERSION,
        "status": "blocked",
        "provider": None,
        "instance_id": None,
        "mode": None,
        "winner_provider": None,
        "winner_launch": None,
        "contenders": records,
        "skipped": skipped_names,
        "terminated_losers": terminated,
        "reason": reason,
    }
