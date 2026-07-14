#!/usr/bin/env python3
"""Control plane for the persistent Isaac warm render worker (--serve lane).

The cold-start problem stops mattering when Isaac boots ONCE and stays resident: the
2026-06-29 GPU runs proved a warm ``--serve`` pod renders a task in ~30s vs ~8min for a
cold boot. This script owns that pod's lifecycle from the operator's machine:

  start    launch ONE serve pod via the parity job (bundles + stages + boots Isaac + loads
           the kitchen once, then polls a presigned job inbox). Writes ``warm_serve_pod.json``
           next to the job manifest so gpu_spend_guard reports the pod as an EXPECTED warm
           worker instead of an anomaly.
  submit   send task scenarios to the running pod (single --task or a scenarios JSON) and
           collect each result from the pod's output channel.
  status   provider-side view of the pod plus the local marker record.
  stop     best-effort stop sentinel through the inbox, then terminate the pod and mark
           the local record terminated.

Claim boundary: lifecycle orchestration only — it does not prove render quality, task
success, or robot readiness (those come from the collected per-task results).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for candidate in (str(SRC_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from blueprint_pipeline.isaac_g1_kitchen_parity_job import (  # noqa: E402
    ISAAC_G1_KITCHEN_PARITY_LANE,
    ISAAC_G1_KITCHEN_PARITY_RESOURCE_PREFIX,
    JOB_MANIFEST_FILENAME,
    _teardown_proof_from_attempt,
    run_isaac_g1_kitchen_parity_job,
)
from blueprint_pipeline.common import write_json  # noqa: E402
from blueprint_pipeline.paid_lane_guard import close_pending_teardown  # noqa: E402
from blueprint_pipeline.paid_provider_lane_lease import (  # noqa: E402
    PaidProviderLaneLeaseSet,
)
from blueprint_pipeline.production_gpu_campaign_budget import (  # noqa: E402
    ProductionGpuCampaignBudget,
)
from blueprint_pipeline.production_gpu_worker_agent import (  # noqa: E402
    POOL_TOKEN_FILE_ENV,
    RUNPOD_GPU_POOL_CLASS,
    _post_json,
    _read_json_record,
    _read_token,
    build_worker_registration_payload,
    run_worker_agent,
)
from blueprint_pipeline.warm_render_server import submit_warm_render_batch  # noqa: E402

WARM_SERVE_MARKER_FILENAME = "warm_serve_pod.json"
MARKER_SCHEMA_VERSION = "warm_serve_pod.v1"
WATCHDOG_EVIDENCE_FILENAME = "production_gpu_warm_watchdog.json"
WATCHDOG_CANCEL_FILENAME = "production_gpu_warm_watchdog.cancel"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _marker_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / WARM_SERVE_MARKER_FILENAME


def read_marker(out_dir: str | Path) -> dict[str, Any]:
    path = _marker_path(out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no {WARM_SERVE_MARKER_FILENAME} under {out_dir} — run start first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("warm_serve_pod.json is not a mapping")
    return payload


def write_marker(out_dir: str | Path, payload: dict[str, Any]) -> Path:
    path = _marker_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path


def _lease_fields(ttl_seconds: float) -> dict[str, str]:
    now = datetime.now(timezone.utc)
    return {
        "heartbeat_at": now.isoformat(),
        "lease_expires_at": (
            now + timedelta(seconds=max(60.0, float(ttl_seconds)))
        ).isoformat(),
    }


def _refresh_marker_lease(out_dir: str | Path) -> dict[str, Any]:
    marker = read_marker(out_dir)
    if marker.get("status") == "serving":
        marker.update(
            _lease_fields(float(marker.get("serve_idle_timeout_s") or 900.0) + 300.0)
        )
        write_marker(out_dir, marker)
    return marker


def launch_teardown_watchdog(
    *,
    out_dir: str | Path,
    hard_ttl_seconds: int,
    campaign_reservation: Mapping[str, Any] | None = None,
    pool_base_url: str | None = None,
    pool_token_file: str | Path | None = None,
) -> dict[str, Any]:
    """Start a process-group-independent provider terminator before allocation."""

    ttl = int(hard_ttl_seconds)
    if not 120 <= ttl <= 10_980:
        raise ValueError("warm_watchdog_hard_ttl_out_of_range")
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = root / WATCHDOG_EVIDENCE_FILENAME
    cancel_path = root / WATCHDOG_CANCEL_FILENAME
    for stale in (evidence_path, cancel_path):
        if stale.exists():
            stale.unlink()
    deadline = time.time() + ttl
    stdout_path = root / "production_gpu_warm_watchdog.stdout.log"
    stderr_path = root / "production_gpu_warm_watchdog.stderr.log"
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        command = [
                sys.executable,
                "-m",
                "blueprint_pipeline.production_gpu_warm_watchdog",
                "--out-dir",
                str(root),
                "--deadline-epoch",
                str(deadline),
            ]
        if campaign_reservation:
            command.extend(
                [
                    "--campaign-budget-ledger",
                    str(campaign_reservation["ledger_path"]),
                    "--campaign-reservation-id",
                    str(campaign_reservation["reservation_id"]),
                ]
            )
        if pool_base_url and pool_token_file:
            command.extend(
                [
                    "--pool-base-url",
                    str(pool_base_url),
                    "--pool-token-file",
                    str(Path(pool_token_file).expanduser().resolve()),
                ]
            )
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
    wait_deadline = time.monotonic() + 10
    evidence: dict[str, Any] = {}
    while time.monotonic() < wait_deadline:
        if process.poll() is not None:
            raise RuntimeError("warm_watchdog_exited_before_arming")
        try:
            value = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence = dict(value) if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            evidence = {}
        if evidence.get("status") == "armed" and evidence.get("pid") == process.pid:
            return evidence
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError("warm_watchdog_did_not_arm")


def reserve_campaign_budget(
    *,
    ledger_path: str | Path,
    hard_ttl_seconds: int,
    max_hourly_rate_usd: float,
    initial_spent_usd: float | None,
    initial_used_gpu_seconds: int | None,
    reservation_id: str | None = None,
) -> dict[str, Any]:
    """Atomically reserve both campaign caps before any provider mutation."""

    path = Path(ledger_path).expanduser().resolve()
    if path.is_file():
        try:
            persisted = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("campaign_budget_ledger_unreadable") from None
        if not isinstance(persisted, Mapping):
            raise ValueError("campaign_budget_ledger_not_object")
        initial_spent_usd = float(cast(Any, persisted.get("initial_spent_usd")))
        initial_used_gpu_seconds = int(
            cast(Any, persisted.get("initial_used_gpu_seconds"))
        )
    elif initial_spent_usd is None or initial_used_gpu_seconds is None:
        raise ValueError("new_campaign_budget_ledger_requires_reconciled_baseline")
    if initial_spent_usd is None or initial_used_gpu_seconds is None:
        raise ValueError("campaign_budget_ledger_baseline_missing")
    ledger = ProductionGpuCampaignBudget(
        path,
        initial_spent_usd=float(cast(Any, initial_spent_usd)),
        initial_used_gpu_seconds=int(cast(Any, initial_used_gpu_seconds)),
    )
    key = str(reservation_id or f"warm-{uuid.uuid4()}")
    reservation = ledger.reserve(
        reservation_id=key,
        gpu_seconds=int(hard_ttl_seconds),
        max_hourly_rate_usd=float(max_hourly_rate_usd),
    )
    return {
        **reservation,
        "ledger_path": str(path),
        "ledger_snapshot": ledger.snapshot(),
    }


def register_production_worker(
    *,
    out_dir: str | Path,
    manifest: Mapping[str, Any],
    endpoint_ref: str,
    pool_base_url: str,
    pool_token_file: str | Path,
    sender: Callable[[str, str, Mapping[str, Any], str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join the three downloaded evidence layers and register the exact release."""

    warm_serve = manifest.get("warm_serve")
    if not isinstance(warm_serve, Mapping):
        raise ValueError("production_registration_warm_serve_missing")
    ready_detail = warm_serve.get("ready_detail")
    if not isinstance(ready_detail, Mapping):
        raise ValueError("production_registration_ready_detail_missing")
    paths = ready_detail.get("registration_evidence_paths")
    if not isinstance(paths, Mapping):
        raise ValueError("production_registration_evidence_paths_missing")
    host = _read_json_record(str(paths.get("host") or ""), label="host")
    cache = _read_json_record(str(paths.get("cache") or ""), label="cache")
    warm = _read_json_record(str(paths.get("warm") or ""), label="warm")
    worker_id = str(warm_serve.get("instance_id") or "").strip()
    resolved_endpoint_ref = endpoint_ref.replace(
        "{worker_id}", urllib.parse.quote(worker_id, safe="")
    )
    payload = build_worker_registration_payload(
        worker_id=worker_id,
        provider="runpod",
        host_image_id=str(host.get("host_image_id") or ""),
        worker_image_ref=str(cache.get("worker_image_ref") or ""),
        gpu_family=RUNPOD_GPU_POOL_CLASS,
        endpoint_ref=resolved_endpoint_ref,
        launch_session_id=str(warm.get("launch_session_id") or ""),
        host_evidence=host,
        cache_evidence=cache,
        warm_evidence=warm,
    )
    registration_path = Path(out_dir) / "production_worker_registration_payload.json"
    write_json(registration_path, payload)
    kwargs: dict[str, Any] = {}
    if sender is not None:
        kwargs["sender"] = sender
    registered = run_worker_agent(
        registration_payload=payload,
        pool_base_url=pool_base_url,
        token=_read_token(pool_token_file),
        once=True,
        **kwargs,
    )
    return {
        "status": registered.get("status"),
        "worker_id": worker_id,
        "registration_payload_path": str(registration_path.resolve()),
        "registration_payload": payload,
    }


def launch_worker_heartbeat_agent(
    *,
    out_dir: str | Path,
    registration: Mapping[str, Any],
    pool_base_url: str,
    pool_token_file: str | Path,
) -> dict[str, Any]:
    payload = registration.get("registration_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("heartbeat_agent_registration_payload_missing")
    evidence = Path(out_dir) / "production_registration_evidence"
    command = [
        sys.executable,
        "-m",
        "blueprint_pipeline.production_gpu_worker_agent",
        "--pool-base-url",
        pool_base_url,
        "--worker-id",
        str(payload["worker_id"]),
        "--provider",
        str(payload["provider"]),
        "--host-image-id",
        str(payload["host_image_id"]),
        "--worker-image-ref",
        str(payload["worker_image_ref"]),
        "--gpu-family",
        str(payload["gpu_family"]),
        "--endpoint-ref",
        str(payload["endpoint_ref"]),
        "--launch-session-id",
        str(payload["agent_evidence"]["launch_session_id"]),
        "--host-evidence",
        str(evidence / "production_host_boot_evidence.json"),
        "--cache-evidence",
        str(evidence / "production_cache_evidence.json"),
        "--warm-evidence",
        str(evidence / "warm_serve_ready.json"),
    ]
    root = Path(out_dir).expanduser().resolve()
    env = dict(os.environ)
    env[POOL_TOKEN_FILE_ENV] = str(Path(pool_token_file).expanduser().resolve())
    with (root / "production_gpu_worker_agent.stdout.log").open("ab") as stdout_handle, (
        root / "production_gpu_worker_agent.stderr.log"
    ).open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
    time.sleep(0.25)
    if process.poll() is not None:
        raise RuntimeError("production_gpu_worker_agent_exited_after_registration")
    return {"status": "monitoring", "pid": process.pid}


def start_warm_worker(
    *,
    out_dir: str | Path,
    kitchen_asset_dir: str | None,
    kitchen_url: str | None,
    provider: str,
    allow_paid: bool,
    warm_candidates: Sequence[str],
    marker_timeout: int,
    serve_idle_timeout_s: float,
    serve_max_jobs: int | None,
    serve_ready_timeout: int,
    scenarios: Sequence[dict[str, Any]] = (),
    production_warmup_before_ready: bool = False,
    teardown_supervisor: Mapping[str, Any] | None = None,
    pool_base_url: str | None = None,
    pool_token_file: str | Path | None = None,
    worker_endpoint_ref: str | None = None,
    registration_sender: Callable[
        [str, str, Mapping[str, Any], str], dict[str, Any]
    ] | None = None,
    heartbeat_launcher: Callable[..., Mapping[str, Any]] = launch_worker_heartbeat_agent,
    image: str | None = None,
    worker_image_manifest_diagnostic: str | Path | None = None,
    job_fn: Callable[..., dict] = run_isaac_g1_kitchen_parity_job,
) -> dict[str, Any]:
    """Launch the serve pod and record it as an expected warm worker."""
    if production_warmup_before_ready and not all(
        (pool_base_url, pool_token_file, worker_endpoint_ref)
    ):
        raise ValueError("production_warm_worker_requires_secure_pool_registration")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = job_fn(
        scenarios=list(scenarios),
        out_dir=out_path,
        kitchen_asset_dir=kitchen_asset_dir,
        kitchen_url=kitchen_url,
        provider=provider,
        allow_paid=allow_paid,
        image=image,
        worker_image_manifest_diagnostic=worker_image_manifest_diagnostic,
        marker_timeout=marker_timeout,
        warm_candidates=tuple(warm_candidates or ()),
        serve=True,
        serve_idle_timeout_s=serve_idle_timeout_s,
        serve_max_jobs=serve_max_jobs,
        serve_ready_timeout=serve_ready_timeout,
        serve_production_warmup_before_ready=production_warmup_before_ready,
        runpod_gpu_types=("NVIDIA L40S",) if production_warmup_before_ready else None,
        serve_teardown_supervisor=teardown_supervisor,
    )
    manifest_path = out_path / JOB_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    result: dict[str, Any] = {
        "status": manifest.get("status"),
        "blockers": manifest.get("blockers") or [],
        "manifest_path": str(manifest_path),
    }
    if teardown_supervisor:
        result["teardown_supervisor"] = dict(teardown_supervisor)
    warm_serve = manifest.get("warm_serve") or {}
    if manifest.get("status") == "serving" and warm_serve.get("instance_id"):
        registration: dict[str, Any] | None = None
        heartbeat_agent: dict[str, Any] | None = None
        if production_warmup_before_ready:
            try:
                registration = register_production_worker(
                    out_dir=out_path,
                    manifest=manifest,
                    endpoint_ref=str(worker_endpoint_ref),
                    pool_base_url=str(pool_base_url),
                    pool_token_file=str(pool_token_file),
                    sender=registration_sender,
                )
                heartbeat_agent = dict(
                    heartbeat_launcher(
                        out_dir=out_path,
                        registration=registration,
                        pool_base_url=str(pool_base_url),
                        pool_token_file=str(pool_token_file),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - watchdog owns bounded cleanup
                blocked_marker = {
                    "schema_version": MARKER_SCHEMA_VERSION,
                    "status": "registration_blocked_teardown_pending",
                    "provider": provider,
                    "pod_id": warm_serve["instance_id"],
                    "manifest_path": str(manifest_path),
                    "pending_teardown_record": warm_serve.get("pending_teardown_record"),
                    "teardown_supervisor": dict(teardown_supervisor or {}),
                    "registration_error_type": type(exc).__name__,
                    "pool_base_url": pool_base_url,
                    "pool_token_file": str(Path(str(pool_token_file)).expanduser().resolve()),
                }
                result.update(
                    status="registration_blocked_teardown_pending",
                    blockers=["production_worker_registration_failed"],
                    registration_error_type=type(exc).__name__,
                    marker_path=str(write_marker(out_path, blocked_marker)),
                    pod_id=warm_serve["instance_id"],
                )
                return result
            result["production_registration"] = registration
            result["heartbeat_agent"] = heartbeat_agent
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "status": "serving",
            "provider": provider,
            "pod_id": warm_serve["instance_id"],
            "started_at": _utc_now_iso(),
            **_lease_fields(float(serve_idle_timeout_s) + 300.0),
            "manifest_path": str(manifest_path),
            "pending_teardown_record": warm_serve.get("pending_teardown_record"),
            "broker_base_url_file": warm_serve.get("broker_base_url_file"),
            "broker_token_file": warm_serve.get("broker_token_file"),
            "transport": "durable_warm_render_broker",
            "single_object_transport_enabled": False,
            "serve_idle_timeout_s": serve_idle_timeout_s,
            "production_warmup_before_ready": production_warmup_before_ready,
            "teardown_supervisor": dict(teardown_supervisor or {}),
            "production_registration": registration,
            "heartbeat_agent": heartbeat_agent,
            "pool_base_url": pool_base_url,
            "pool_token_file": (
                str(Path(pool_token_file).expanduser().resolve())
                if pool_token_file
                else None
            ),
            "note": (
                "Expected persistent warm render worker — gpu_spend_guard tags this pod "
                "instead of treating it as an anomaly. Stop it with "
                "'run_warm_render_worker.py stop' when done."
            ),
        }
        result["marker_path"] = str(write_marker(out_path, marker))
        result["pod_id"] = warm_serve["instance_id"]
    elif teardown_supervisor:
        (out_path / WATCHDOG_CANCEL_FILENAME).write_text(
            "launch_did_not_enter_serving_state\n", encoding="utf-8"
        )
    return result


def submit_tasks(
    *,
    out_dir: str | Path,
    tasks: Sequence[str],
    scenarios_json: str | None,
    timeout_s: float,
    interval_s: float,
    stop_after: bool = False,
    submit_fn: Callable[..., dict] = submit_warm_render_batch,
) -> dict[str, Any]:
    """Submit one or more task scenarios to the serving pod and collect results."""
    marker = _refresh_marker_lease(out_dir)
    if marker.get("status") != "serving":
        raise RuntimeError(f"warm worker status is {marker.get('status')!r}, not 'serving'")
    if scenarios_json:
        scenarios_path = Path(scenarios_json)
    else:
        scenarios = [
            {
                "scenario_id": f"warm_task_{index}",
                "instruction": task,
                "task": task,
            }
            for index, task in enumerate(tasks, start=1)
        ]
        if not scenarios:
            raise ValueError("no --task given and no --scenarios-json provided")
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".json", prefix="warm_scenarios_", delete=False, encoding="utf-8"
        )
        with tmp:
            json.dump({"scenarios": scenarios}, tmp)
        scenarios_path = Path(tmp.name)
    result = submit_fn(
        manifest_path=marker["manifest_path"],
        scenarios_path=scenarios_path,
        out_dir=Path(out_dir) / "warm_results",
        timeout_s=timeout_s,
        interval_s=interval_s,
        stop_after=stop_after,
    )
    if not stop_after:
        _refresh_marker_lease(out_dir)
    return result


def stop_warm_worker(
    *,
    out_dir: str | Path,
    provider_factory: Callable[[str], Any] | None = None,
    pool_sender: Callable[
        [str, str, Mapping[str, Any], str], dict[str, Any]
    ] = _post_json,
) -> dict[str, Any]:
    """Terminate the serve pod and mark the local record terminated.

    Termination through the provider API is authoritative; the inbox stop sentinel is not
    attempted here because a fresh client's sequence numbers are stale to the pod by design.
    """
    marker = read_marker(out_dir)
    pod_id = str(marker.get("pod_id") or "")
    provider_name = str(marker.get("provider") or "runpod")
    if not pod_id:
        raise ValueError("warm_serve_pod.json has no pod_id")
    if provider_factory is None:
        from blueprint_pipeline.gpu_render_providers import get_render_provider
        provider_factory = get_render_provider
    provider = provider_factory(provider_name)
    pool_quarantine: dict[str, Any] = {"status": "not_configured"}
    pool_base_url = str(marker.get("pool_base_url") or "").strip()
    pool_token_file = str(marker.get("pool_token_file") or "").strip()
    if pool_base_url and pool_token_file:
        try:
            pool_quarantine = pool_sender(
                pool_base_url,
                f"/v1/workers/{pod_id}/quarantine",
                {"reason": "operator_provider_teardown"},
                _read_token(pool_token_file),
            )
            pool_quarantine["status"] = (
                "quarantined"
                if pool_quarantine.get("state") == "quarantined"
                else "blocked"
            )
        except Exception as exc:  # noqa: BLE001
            pool_quarantine = {"status": "blocked", "error_type": type(exc).__name__}
    try:
        teardown = provider.terminate(pod_id)
    except Exception as exc:  # noqa: BLE001 - retain a retryable blocked marker
        teardown = {
            "status": "terminate_failed",
            "error_type": type(exc).__name__,
            "raw_provider_response_recorded": False,
        }
    proof = _teardown_proof_from_attempt(
        provider=provider,
        instance_id=pod_id,
        teardown=teardown if isinstance(teardown, dict) else {},
        action="terminate",
    )
    pending_path = str(marker.get("pending_teardown_record") or "")
    pending_close = {"status": "not_applicable"}
    if pending_path:
        try:
            pending_close = close_pending_teardown(pending_path, proof)
        except Exception as exc:  # noqa: BLE001 - a failed close keeps the lane blocked
            pending_close = {
                "status": "close_failed",
                "error_type": type(exc).__name__,
            }
    lease_set = PaidProviderLaneLeaseSet(
        providers={provider_name: provider},
        lane=ISAAC_G1_KITCHEN_PARITY_LANE,
        job_dir=str(Path(out_dir) / "warm_stop"),
        resource_name_prefix=ISAAC_G1_KITCHEN_PARITY_RESOURCE_PREFIX,
    )
    lease_acquire = lease_set.acquire()
    lease_release = (
        lease_set.release("warm_serve_terminal_stop", provider_mutation_started=True)
        if lease_acquire.get("status") == "acquired"
        else None
    )
    lease_released = bool(
        lease_release
        and lease_release.get("results")
        and all(
            item.get("status") in {"released", "already_released"}
            for item in lease_release["results"]
        )
    )
    pending_closed = pending_close.get("status") in {"closed", "not_applicable"}
    terminal = bool(
        str(proof.get("status") or "").upper() == "PASS"
        and pending_closed
        and lease_released
        and pool_quarantine.get("status") in {"quarantined", "not_configured"}
    )
    completed_at = _utc_now_iso()
    marker["status"] = "terminated" if terminal else "teardown_blocked"
    marker["terminated_at"] = completed_at if terminal else None
    marker["heartbeat_at"] = completed_at
    marker["lease_expires_at"] = completed_at
    marker["teardown"] = teardown
    marker["teardown_proof"] = proof
    marker["pending_teardown_close"] = pending_close
    marker["paid_provider_lane_lease"] = lease_set.summary
    marker["pool_quarantine"] = pool_quarantine
    write_marker(out_dir, marker)
    return {
        "status": marker["status"],
        "pod_id": pod_id,
        "teardown": teardown,
        "teardown_proof": proof,
        "pending_teardown_close": pending_close,
        "paid_provider_lane_lease": lease_set.summary,
        "pool_quarantine": pool_quarantine,
    }


def worker_status(
    *,
    out_dir: str | Path,
    provider_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    marker = read_marker(out_dir)
    status: dict[str, Any] = {"marker": marker}
    pod_id = str(marker.get("pod_id") or "")
    if pod_id and marker.get("status") == "serving":
        if provider_factory is None:
            from blueprint_pipeline.gpu_render_providers import get_render_provider
            provider_factory = get_render_provider
        try:
            status["provider_view"] = provider_factory(
                str(marker.get("provider") or "runpod")
            ).inspect(pod_id)
            status["marker"] = _refresh_marker_lease(out_dir)
        except Exception as exc:  # noqa: BLE001 - status must not crash on a dead pod
            status["provider_view"] = {"status": "inspect_failed", "error": repr(exc)[:200]}
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="launch the persistent serve pod")
    start.add_argument("--out-dir", required=True)
    start.add_argument("--kitchen-asset-dir", default=None)
    start.add_argument("--kitchen-url", default=None)
    start.add_argument("--provider", default="runpod")
    start.add_argument("--allow-paid", action="store_true")
    start.add_argument("--image", default=None)
    start.add_argument(
        "--worker-image-manifest-diagnostic",
        default=None,
        help=(
            "registry manifest diagnostic JSON for the exact digest-pinned worker image; "
            "required for paid digest-pinned starts"
        ),
    )
    start.add_argument("--warm-candidate", action="append", default=[])
    start.add_argument("--marker-timeout", type=int, default=900)
    start.add_argument("--serve-idle-timeout", type=float, default=1800.0,
                       help=("runner idle bound only; paid serve additionally requires a "
                             "provider teardown supervisor"))
    start.add_argument("--serve-max-jobs", type=int, default=None)
    start.add_argument("--serve-ready-timeout", type=int, default=1800)
    start.add_argument(
        "--production",
        action="store_true",
        help="require same-session renderer/policy warmup and production evidence",
    )
    start.add_argument(
        "--warmup-scenarios-json",
        default=None,
        help="required with --production; first scenario is executed before readiness",
    )
    start.add_argument(
        "--hard-ttl-seconds",
        type=int,
        default=0,
        help="provider API teardown deadline; required for every paid warm start",
    )
    start.add_argument(
        "--campaign-budget-ledger",
        default=None,
        help="durable dual-cap ledger; required for every paid warm start",
    )
    start.add_argument("--campaign-initial-spent-usd", type=float, default=None)
    start.add_argument("--campaign-initial-used-gpu-seconds", type=int, default=None)
    start.add_argument("--campaign-reservation-id", default=None)
    start.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
    start.add_argument("--pool-base-url", default=None)
    start.add_argument("--pool-token-file", default=None)
    start.add_argument(
        "--worker-endpoint-ref",
        default=None,
        help="credential-free HTTPS broker route for this worker",
    )

    submit = sub.add_parser("submit", help="submit tasks to the serving pod")
    submit.add_argument("--out-dir", required=True, help="the start command's --out-dir")
    submit.add_argument("--task", action="append", default=[],
                        help="free-form task string (repeatable)")
    submit.add_argument("--scenarios-json", default=None,
                        help="JSON list / {scenarios:[...]} file instead of --task")
    submit.add_argument("--timeout", type=float, default=900.0)
    submit.add_argument("--interval", type=float, default=5.0)
    submit.add_argument("--stop-after", action="store_true")

    status = sub.add_parser("status", help="local marker + provider view of the pod")
    status.add_argument("--out-dir", required=True)

    stop = sub.add_parser("stop", help="terminate the serve pod")
    stop.add_argument("--out-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "start":
        scenarios: list[dict[str, Any]] = []
        if args.warmup_scenarios_json:
            raw_scenarios = json.loads(
                Path(args.warmup_scenarios_json).read_text(encoding="utf-8")
            )
            if isinstance(raw_scenarios, dict):
                raw_scenarios = raw_scenarios.get("scenarios") or []
            if not isinstance(raw_scenarios, list) or not all(
                isinstance(row, dict) for row in raw_scenarios
            ):
                raise SystemExit("warmup scenarios must be a JSON list of objects")
            scenarios = [dict(row) for row in raw_scenarios]
        if args.production and not scenarios:
            raise SystemExit("--production requires --warmup-scenarios-json")
        if args.production and not all(
            (args.pool_base_url, args.pool_token_file, args.worker_endpoint_ref)
        ):
            raise SystemExit(
                "--production requires --pool-base-url, --pool-token-file, "
                "and --worker-endpoint-ref"
            )
        teardown_supervisor = None
        if args.allow_paid:
            if args.hard_ttl_seconds <= 0:
                raise SystemExit("paid warm start requires --hard-ttl-seconds")
            if not args.campaign_budget_ledger:
                raise SystemExit("paid warm start requires --campaign-budget-ledger")
            try:
                campaign_reservation = reserve_campaign_budget(
                    ledger_path=args.campaign_budget_ledger,
                    hard_ttl_seconds=args.hard_ttl_seconds,
                    max_hourly_rate_usd=args.max_hourly_rate_usd,
                    initial_spent_usd=args.campaign_initial_spent_usd,
                    initial_used_gpu_seconds=args.campaign_initial_used_gpu_seconds,
                    reservation_id=args.campaign_reservation_id,
                )
            except (ValueError, RuntimeError) as exc:
                raise SystemExit(f"campaign budget admission blocked: {exc}") from exc
            teardown_supervisor = launch_teardown_watchdog(
                out_dir=args.out_dir,
                hard_ttl_seconds=args.hard_ttl_seconds,
                campaign_reservation=campaign_reservation,
                pool_base_url=args.pool_base_url,
                pool_token_file=args.pool_token_file,
            )
        result = start_warm_worker(
            out_dir=args.out_dir,
            kitchen_asset_dir=args.kitchen_asset_dir,
            kitchen_url=args.kitchen_url,
            provider=args.provider,
            allow_paid=args.allow_paid,
            image=args.image,
            worker_image_manifest_diagnostic=args.worker_image_manifest_diagnostic,
            warm_candidates=args.warm_candidate,
            marker_timeout=args.marker_timeout,
            serve_idle_timeout_s=args.serve_idle_timeout,
            serve_max_jobs=args.serve_max_jobs,
            serve_ready_timeout=args.serve_ready_timeout,
            scenarios=scenarios,
            production_warmup_before_ready=args.production,
            teardown_supervisor=teardown_supervisor,
            pool_base_url=args.pool_base_url,
            pool_token_file=args.pool_token_file,
            worker_endpoint_ref=args.worker_endpoint_ref,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") in ("serving", "prepared") else 1
    if args.command == "submit":
        result = submit_tasks(
            out_dir=args.out_dir,
            tasks=args.task,
            scenarios_json=args.scenarios_json,
            timeout_s=args.timeout,
            interval_s=args.interval,
            stop_after=args.stop_after,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") == "completed" else 1
    if args.command == "status":
        print(json.dumps(worker_status(out_dir=args.out_dir), indent=2, default=str))
        return 0
    if args.command == "stop":
        print(json.dumps(stop_warm_worker(out_dir=args.out_dir), indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
