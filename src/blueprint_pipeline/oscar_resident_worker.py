"""Resident OSCAR worker: load weights once, serve many closed-loop steps.

The per-step OSCAR path spawns a full ``torch.distributed.run`` invocation for
every closed-loop step, so a 2B checkpoint is read from disk, moved to device,
and torn down once per generated observation. For a 300-step rollout that is 300
model loads to produce 300 frames.

That is not merely wasteful, it is measurement-limiting. Rank-fidelity
confidence is bought with rollouts, and rollouts are bought with throughput, so
generation cost per step sets how many policies a qualification campaign can
afford to cover -- which is the binding constraint on every correlation interval
the platform can publish.

This module keeps one worker process alive across a rollout and speaks a
line-delimited JSON protocol to it:

* the worker loads the checkpoint once, then emits a ``ready`` line carrying its
  load time and device identity;
* each ``generate`` request returns the produced clip plus its own warm timing;
* the client records cold-start and warm-step latency separately, so the report
  shows what residency actually bought rather than asserting it.

Failure handling is deliberately loud. A dead or desynchronised worker fails the
step closed rather than silently falling back to per-step spawning, because a
silent fallback would restore the cold-start cost while continuing to report
resident-path timings. Restarts are permitted only when explicitly budgeted and
are counted in the report, so a crash-loop shows up as a restart count instead of
hiding inside an average.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable


REQUEST_SCHEMA_VERSION = "oscar_resident_worker_request.v1"
RESPONSE_SCHEMA_VERSION = "oscar_resident_worker_response.v1"
READY_SCHEMA_VERSION = "oscar_resident_worker_ready.v1"
THROUGHPUT_SCHEMA_VERSION = "wam_generation_throughput.v1"

DEFAULT_STARTUP_TIMEOUT_SECONDS = 1800.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_SHUTDOWN_GRACE_SECONDS = 15.0


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[position], 6)


class ResidentWorkerError(RuntimeError):
    """Raised when the worker dies or violates the protocol."""


class ResidentOscarWorker:
    """Client for a long-lived OSCAR generation process."""

    def __init__(
        self,
        *,
        argv: Sequence[str],
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
        max_restarts: int = 0,
        require_gpu_residency: bool = False,
        popen: Callable[..., Any] = subprocess.Popen,
        process_group_signal: Callable[[int, int], None] = os.killpg,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._argv = list(argv)
        self._cwd = str(cwd) if cwd is not None else None
        self._env = dict(env) if env is not None else None
        self._startup_timeout = max(1.0, float(startup_timeout_seconds))
        self._request_timeout = max(1.0, float(request_timeout_seconds))
        self._shutdown_grace = max(0.1, float(shutdown_grace_seconds))
        self._max_restarts = max(0, int(max_restarts))
        self._require_gpu_residency = bool(require_gpu_residency)
        self._popen = popen
        self._process_group_signal = process_group_signal
        self._monotonic = monotonic

        self._process: Any = None
        self._reader: threading.Thread | None = None
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._ready: dict[str, Any] = {}
        self._alive = False
        self._request_counter = 0

        self.cold_start_seconds: float | None = None
        self.cold_start_count = 0
        self.restart_count = 0
        self.warm_step_seconds: list[float] = []
        self.failures: list[dict[str, Any]] = []
        self._seen_runtime_result_ids: set[str] = set()

    # -- lifecycle ---------------------------------------------------------

    def _drain_reader(self) -> None:
        while True:
            try:
                line = self._process.stdout.readline()
            except Exception:
                line = ""
            if not line:
                self._lines.put(None)
                return
            self._lines.put(line)

    def _read_line(self, timeout: float) -> str:
        deadline = self._monotonic() + timeout
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise ResidentWorkerError("oscar_resident_worker_response_timeout")
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                raise ResidentWorkerError("oscar_resident_worker_stream_closed")
            text = line.strip()
            if text:
                return text

    def start(self) -> dict[str, Any]:
        """Spawn the worker and block until it reports readiness."""

        started = self._monotonic()
        self._process = self._popen(
            self._argv,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._drain_reader, daemon=True)
        self._reader.start()

        payload = json.loads(self._read_line(self._startup_timeout))
        if payload.get("schema_version") != READY_SCHEMA_VERSION:
            raise ResidentWorkerError("oscar_resident_worker_ready_schema_invalid")
        if payload.get("status") != "ready":
            raise ResidentWorkerError("oscar_resident_worker_not_ready")
        if self._require_gpu_residency and not _string(payload.get("cuda_device_name")):
            # Residency on the resident path is reported by the worker itself
            # rather than sampled per subprocess, so an absent device identity
            # must fail closed rather than be assumed.
            raise ResidentWorkerError("oscar_resident_worker_gpu_residency_unproven")

        self._ready = dict(payload)
        self._alive = True
        self.cold_start_count += 1
        observed = self._monotonic() - started
        reported = _number(payload.get("model_load_seconds"))
        self.cold_start_seconds = round(reported if reported is not None else observed, 6)
        return dict(self._ready)

    def _restart(self) -> None:
        if self.restart_count >= self._max_restarts:
            raise ResidentWorkerError("oscar_resident_worker_restart_budget_exhausted")
        self.restart_count += 1
        self.close(record_failure=False)
        self.start()

    # -- generation --------------------------------------------------------

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run one warm generation on the resident worker."""

        if not self._alive:
            raise ResidentWorkerError("oscar_resident_worker_not_started")
        self._request_counter += 1
        request_id = f"req-{self._request_counter:06d}"
        payload = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "op": "generate",
            **{key: value for key, value in dict(request).items() if key != "op"},
        }
        started = self._monotonic()
        try:
            self._process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            self._process.stdin.flush()
            response = json.loads(self._read_line(self._request_timeout))
        except ResidentWorkerError:
            self._alive = False
            self.failures.append({"request_id": request_id, "reason": "worker_unavailable"})
            if self._max_restarts:
                self._restart()
            raise
        except (BrokenPipeError, OSError, ValueError) as error:
            self._alive = False
            self.failures.append({"request_id": request_id, "reason": type(error).__name__})
            raise ResidentWorkerError("oscar_resident_worker_transport_failed") from error
        elapsed = self._monotonic() - started

        if response.get("schema_version") != RESPONSE_SCHEMA_VERSION:
            raise ResidentWorkerError("oscar_resident_worker_response_schema_invalid")
        if _string(response.get("request_id")) != request_id:
            # A mismatched id means the stream has desynchronised; continuing
            # would attribute one step's output to another step's action.
            self._alive = False
            raise ResidentWorkerError("oscar_resident_worker_response_out_of_order")
        runtime_result_id = _string(response.get("runtime_result_id"))
        if runtime_result_id:
            if runtime_result_id in self._seen_runtime_result_ids:
                self._alive = False
                raise ResidentWorkerError("oscar_resident_worker_runtime_result_replayed")
            self._seen_runtime_result_ids.add(runtime_result_id)

        self.warm_step_seconds.append(round(elapsed, 6))
        result = dict(response)
        result["client_elapsed_seconds"] = round(elapsed, 6)
        return result

    # -- teardown ----------------------------------------------------------

    def close(self, *, record_failure: bool = True) -> None:
        process = self._process
        if process is None:
            return
        self._alive = False
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write(
                        json.dumps(
                            {"schema_version": REQUEST_SCHEMA_VERSION, "op": "shutdown"},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    # The worker already exited or closed its stdin; the signal
                    # and wait path below still guarantees teardown.
                    pass
            try:
                process.wait(timeout=self._shutdown_grace)
            except subprocess.TimeoutExpired:
                group = getattr(process, "pid", None)
                if isinstance(group, int) and group > 0:
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        try:
                            self._process_group_signal(group, sig)
                        except (ProcessLookupError, OSError):
                            break
                        try:
                            process.wait(timeout=self._shutdown_grace)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                if record_failure:
                    self.failures.append({"reason": "shutdown_required_signal"})
        finally:
            self._process = None

    def __enter__(self) -> "ResidentOscarWorker":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- reporting ---------------------------------------------------------

    def close_and_report(self, output_dir: str | Path) -> dict[str, Any]:
        """Write the throughput report, then tear the worker down.

        The report is written even when the rollout failed: a run that died
        halfway is exactly when the per-step timings are worth reading.
        """

        report = self.throughput_report()
        try:
            path = Path(output_dir).expanduser() / "oscar_resident_worker_throughput.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        finally:
            self.close()
        return report

    def throughput_report(self) -> dict[str, Any]:
        """What residency actually bought, in measured seconds."""

        warm = list(self.warm_step_seconds)
        warm_mean = round(sum(warm) / len(warm), 6) if warm else None
        cold = self.cold_start_seconds
        # The per-step path pays the load on every step, so its cost model is
        # steps * (load + generate); the resident path pays load once.
        per_step_equivalent = (
            round(cold + warm_mean, 6) if cold is not None and warm_mean is not None else None
        )
        speedup = (
            round(per_step_equivalent / warm_mean, 4)
            if per_step_equivalent and warm_mean and warm_mean > 0
            else None
        )
        return {
            "schema_version": THROUGHPUT_SCHEMA_VERSION,
            "mode": "resident_worker",
            "cold_start_seconds": cold,
            "cold_start_count": self.cold_start_count,
            "restart_count": self.restart_count,
            "warm_step_count": len(warm),
            "warm_step_seconds_mean": warm_mean,
            "warm_step_seconds_p50": _percentile(warm, 0.5),
            "warm_step_seconds_p95": _percentile(warm, 0.95),
            "warm_step_seconds_min": round(min(warm), 6) if warm else None,
            "warm_step_seconds_max": round(max(warm), 6) if warm else None,
            "per_step_spawn_equivalent_seconds": per_step_equivalent,
            "estimated_speedup_vs_per_step_spawn": speedup,
            "steps_per_hour": (
                round(3600.0 / warm_mean, 3) if warm_mean and warm_mean > 0 else None
            ),
            "worker_identity": {
                "checkpoint_sha256": _string(self._ready.get("checkpoint_sha256")) or None,
                "cuda_device_name": _string(self._ready.get("cuda_device_name")) or None,
                "worker_pid": getattr(self._process, "pid", None),
            },
            "failures": list(self.failures),
            "claim_boundary": {
                "throughput_is_not_generation_quality": True,
                "throughput_is_not_task_success": True,
                "speedup_is_measured_against_a_modelled_spawn_cost": True,
            },
        }


def build_resident_worker_argv(
    *,
    python: str,
    oscar_repo: str | Path,
    checkpoint: str | Path,
    num_steps: int,
    guidance: float,
    height: int,
    width: int,
    fps: float,
) -> list[str]:
    """Argv for the resident worker entrypoint.

    Deliberately not ``torch.distributed.run``: a single-process resident worker
    has no launcher to re-elect on every step, which is the cost being removed.
    """

    return [
        python,
        "-m",
        "blueprint_pipeline.oscar_resident_worker_main",
        "--oscar-repo",
        str(Path(oscar_repo).expanduser()),
        "--checkpoint",
        str(checkpoint),
        "--num-steps",
        str(int(num_steps)),
        "--guidance",
        str(float(guidance)),
        "--height",
        str(int(height)),
        "--width",
        str(int(width)),
        "--fps",
        str(float(fps)),
    ]


def start_resident_oscar_generate_from_args(
    args: Any,
    *,
    python: str,
    extract_next_frame: Callable[[Path, Path], Path | None],
    build_skeleton_video: Callable[[Sequence[Mapping[str, Any]], Path], Path | None] | None = None,
    require_gpu_residency: bool = True,
) -> tuple["ResidentOscarWorker", Callable[[Mapping[str, Any]], dict[str, Any]]]:
    """Start a resident worker from parsed closed-loop CLI arguments.

    Returns the worker alongside its generate callable so the caller can tear it
    down and write its throughput report; the worker holds the GPU for the whole
    rollout, so ownership must be explicit rather than implied.
    """

    worker = ResidentOscarWorker(
        argv=build_resident_worker_argv(
            python=python,
            oscar_repo=args.oscar_repo,
            checkpoint=args.checkpoint,
            num_steps=int(args.oscar_num_steps),
            guidance=float(args.oscar_guidance),
            height=int(args.oscar_height),
            width=int(args.oscar_width),
            fps=float(args.oscar_fps),
        ),
        cwd=str(Path(args.oscar_repo).expanduser()),
        env=os.environ.copy(),
        request_timeout_seconds=float(args.provider_timeout_seconds),
        max_restarts=int(args.oscar_resident_worker_max_restarts),
        require_gpu_residency=require_gpu_residency,
    )
    try:
        worker.start()
    except BaseException:
        # start() spawns the process before it validates the ready handshake, so
        # a not-ready payload, startup timeout or missing GPU residency would
        # otherwise leave an OSCAR process alive holding the GPU while the
        # caller unwinds past its teardown path.
        worker.close()
        raise
    generate = make_resident_oscar_generate(
        worker=worker,
        build_skeleton_video=build_skeleton_video,
        extract_next_frame=extract_next_frame,
    )
    return worker, generate


def make_resident_oscar_generate(
    *,
    worker: ResidentOscarWorker,
    extract_next_frame: Callable[[Path, Path], Path | None],
    build_skeleton_video: Callable[[Sequence[Mapping[str, Any]], Path], Path | None] | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Adapt a resident worker to the per-step ``oscar_generate`` contract.

    Returns the same response shape as the spawning path
    (``status``/``blockers``/``generated_frame_path``/``generated_video_path``/
    log paths), so it is a drop-in for ``make_oscar_per_step_wam_backend``.
    """

    def _generate(request: Mapping[str, Any]) -> dict[str, Any]:
        out_dir = Path(_string(request.get("output_dir"))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video = out_dir / "oscar_next_observation.mp4"
        stdout_log = out_dir / "oscar_resident_stdout.log"
        stderr_log = out_dir / "oscar_resident_stderr.log"

        landmarks = request.get("skeleton_landmarks") or []
        skeleton_trace_rows = request.get("skeleton_trace_rows") or []
        skeleton_input = skeleton_trace_rows or landmarks
        skeleton_video = (
            build_skeleton_video(skeleton_input, out_dir) if build_skeleton_video else None
        )
        if build_skeleton_video is not None and skeleton_video is None:
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text(
                "OSCAR resident inference skipped: projected skeleton conditioning "
                "is unavailable.\n",
                encoding="utf-8",
            )
            return {
                "status": "blocked",
                "blockers": ["oscar_per_step_projected_skeleton_conditioning_unavailable"],
                "generated_frame_path": "",
                "generated_video_path": "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }

        try:
            response = worker.generate(
                {
                    "reference_frame_path": _string(request.get("reference_frame_path")),
                    "task_prompt": _string(request.get("task_prompt")),
                    "num_frames": int(request.get("num_frames") or 8),
                    "seed": int(request.get("seed") or 42),
                    "output_video": str(output_video),
                    "skeleton_video": str(skeleton_video) if skeleton_video else "",
                }
            )
        except ResidentWorkerError as error:
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text(f"{error}\n", encoding="utf-8")
            return {
                "status": "blocked",
                "blockers": [f"oscar_resident_worker_failed:{error}"],
                "generated_frame_path": "",
                "generated_video_path": str(output_video) if output_video.is_file() else "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }

        stdout_log.write_text(_string(response.get("stdout_tail")), encoding="utf-8")
        stderr_log.write_text(_string(response.get("stderr_tail")), encoding="utf-8")

        blockers = [
            _string(item) for item in response.get("blockers", []) or [] if _string(item)
        ]
        if _string(response.get("status")) != "ok" or blockers:
            return {
                "status": "blocked",
                "blockers": sorted(set(blockers or ["oscar_resident_worker_generation_failed"])),
                "generated_frame_path": "",
                "generated_video_path": str(output_video) if output_video.is_file() else "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }
        if not output_video.is_file():
            return {
                "status": "blocked",
                "blockers": ["oscar_resident_worker_output_missing"],
                "generated_frame_path": "",
                "generated_video_path": "",
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }

        next_frame = extract_next_frame(output_video, out_dir)
        if next_frame is None:
            return {
                "status": "blocked",
                "blockers": ["oscar_per_step_next_frame_extraction_failed"],
                "generated_frame_path": "",
                "generated_video_path": str(output_video),
                "stdout_log_path": str(stdout_log),
                "stderr_log_path": str(stderr_log),
            }
        return {
            "status": "completed",
            "blockers": [],
            "generated_frame_path": str(next_frame),
            "generated_video_path": str(output_video),
            "stdout_log_path": str(stdout_log),
            "stderr_log_path": str(stderr_log),
            "warm_generate_seconds": response.get("generate_seconds"),
            "client_elapsed_seconds": response.get("client_elapsed_seconds"),
            "runtime_result_id": response.get("runtime_result_id"),
        }

    return _generate
