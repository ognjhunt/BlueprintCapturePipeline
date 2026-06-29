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
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


@dataclass
class WarmJob:
    """One render request handed to the warm worker. ``stop=True`` is the shutdown sentinel."""

    request_id: str
    scenario: dict[str, Any] = field(default_factory=dict)
    stop: bool = False


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

        job = job_source.poll()
        if job is None:
            if clock() - last_activity >= idle_timeout_s:
                _log(f"warm serve loop: idle {idle_timeout_s}s elapsed; exiting after {served} job(s)")
                return {"jobs_served": served, "exit_reason": "idle_timeout"}
            sleep(poll_interval_s)
            continue

        if job.stop:
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
        job_source.publish_result(job.request_id, result)
        served += 1
        last_activity = clock()


class FileJobSource:
    """A :class:`JobSource` backed by a local directory tree — for tests and shared-volume transport.

    Jobs are dropped as ``jobs/<seq>_<request_id>.json`` (FIFO by zero-padded sequence, no clock/random
    needed) and claimed by deletion on :meth:`poll`; results land in ``results/<request_id>.json``.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _next_seq(self) -> str:
        return f"{len(list(self.jobs_dir.glob('*.json'))):06d}"

    def submit(self, request_id: str, scenario: dict[str, Any]) -> Path:
        path = self.jobs_dir / f"{self._next_seq()}_{request_id}.json"
        path.write_text(json.dumps({"request_id": request_id, "scenario": scenario, "stop": False}))
        return path

    def submit_stop(self) -> Path:
        path = self.jobs_dir / f"{self._next_seq()}_stop.json"
        path.write_text(json.dumps({"request_id": "stop", "scenario": {}, "stop": True}))
        return path

    def poll(self) -> Optional[WarmJob]:
        files = sorted(self.jobs_dir.glob("*.json"))
        if not files:
            return None
        path = files[0]
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a half-written job file: drop it and move on
            path.unlink(missing_ok=True)
            return None
        path.unlink(missing_ok=True)  # claim
        return WarmJob(
            request_id=str(payload.get("request_id") or path.stem),
            scenario=dict(payload.get("scenario") or {}),
            stop=bool(payload.get("stop")),
        )

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        (self.results_dir / f"{request_id}.json").write_text(json.dumps(result, indent=2))

    def collect_result(self, request_id: str) -> Optional[dict[str, Any]]:
        path = self.results_dir / f"{request_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())


def _http_get_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    return urllib.request.urlopen(url, timeout=timeout).read()


def _http_put_bytes(url: str, data: bytes, *, timeout: float = 60.0) -> None:
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=timeout).read()


class SignedUrlJobSource:
    """Pod-side :class:`JobSource` over presigned object-store URLs.

    The control plane writes the next job to a single inbox key (one presigned PUT it holds); the pod
    polls that key via a presigned GET and claims jobs by monotonic ``seq`` (so the same job is never
    re-served). Results ride the EXISTING worker output channel: :meth:`publish_result` writes them
    into the pod's out dir (``warm_results/<request_id>.json``), which the worker's heartbeat already
    uploads — so no second presigned channel is needed.
    """

    def __init__(self, inbox_get_url: str, out_dir: Path | str, *,
                 http_get: Callable[[str], bytes] = _http_get_bytes) -> None:
        self.inbox_get_url = inbox_get_url
        self.out_dir = Path(out_dir)
        self.results_dir = self.out_dir / "warm_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._http_get = http_get
        # The inbox is seeded with seq=0 at presign time; start at 0 so that seed is NOT claimed as a
        # job (the control plane's first real submit is seq=1).
        self._last_seq = 0

    def poll(self) -> Optional[WarmJob]:
        try:
            raw = self._http_get(self.inbox_get_url)
        except Exception:  # noqa: BLE001 - empty/absent inbox (404) == no job yet
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:  # noqa: BLE001 - a partial write: treat as no job
            return None
        seq = int(payload.get("seq", -1))
        if seq <= self._last_seq:
            return None
        self._last_seq = seq
        return WarmJob(
            request_id=str(payload.get("request_id") or seq),
            scenario=dict(payload.get("scenario") or {}),
            stop=bool(payload.get("stop")),
        )

    def publish_result(self, request_id: str, result: dict[str, Any]) -> None:
        (self.results_dir / f"{request_id}.json").write_text(json.dumps(result, indent=2))


class WarmPoolClient:
    """Control-plane client for a warm ``--serve`` pod: submit task jobs + collect their results.

    ``submit`` PUTs a job (monotonic ``seq``) to the inbox key the pod polls; ``poll_result`` reads the
    pod's continuously-uploaded output zip (the existing output GET url) and returns the job's result
    once the pod has written it. Keeping the pod RUNNING between submits is what makes reruns seconds —
    the caller is responsible for the pod's lifecycle (it is not torn down here).
    """

    def __init__(self, inbox_put_url: str, output_get_url: str, *,
                 http_put: Callable[[str, bytes], None] = _http_put_bytes,
                 http_get: Callable[[str], bytes] = _http_get_bytes) -> None:
        self.inbox_put_url = inbox_put_url
        self.output_get_url = output_get_url
        self._http_put = http_put
        self._http_get = http_get
        self._seq = 0

    def submit(self, scenario: dict[str, Any], request_id: Optional[str] = None) -> str:
        self._seq += 1
        rid = request_id or f"job-{self._seq}"
        payload = {"seq": self._seq, "request_id": rid, "scenario": scenario, "stop": False}
        self._http_put(self.inbox_put_url, json.dumps(payload).encode())
        return rid

    def submit_stop(self) -> None:
        self._seq += 1
        self._http_put(self.inbox_put_url,
                       json.dumps({"seq": self._seq, "request_id": "stop", "stop": True}).encode())

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
                            return json.loads(z.read(key).decode())
            except Exception:  # noqa: BLE001 - output zip not posted yet / mid-upload: retry
                pass
            sleep(interval_s)
        return None
