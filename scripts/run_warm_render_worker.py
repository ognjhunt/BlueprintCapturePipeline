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
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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
from blueprint_pipeline.warm_render_server import submit_warm_render_batch  # noqa: E402

WARM_SERVE_MARKER_FILENAME = "warm_serve_pod.json"
MARKER_SCHEMA_VERSION = "warm_serve_pod.v1"


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
    image: str | None = None,
    worker_image_manifest_diagnostic: str | Path | None = None,
    job_fn: Callable[..., dict] = run_isaac_g1_kitchen_parity_job,
) -> dict[str, Any]:
    """Launch the serve pod and record it as an expected warm worker."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest = job_fn(
        scenarios=[],
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
    )
    manifest_path = out_path / JOB_MANIFEST_FILENAME
    write_json(manifest_path, manifest)
    result: dict[str, Any] = {
        "status": manifest.get("status"),
        "blockers": manifest.get("blockers") or [],
        "manifest_path": str(manifest_path),
    }
    warm_serve = manifest.get("warm_serve") or {}
    if manifest.get("status") == "serving" and warm_serve.get("instance_id"):
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
            "note": (
                "Expected persistent warm render worker — gpu_spend_guard tags this pod "
                "instead of treating it as an anomaly. Stop it with "
                "'run_warm_render_worker.py stop' when done."
            ),
        }
        result["marker_path"] = str(write_marker(out_path, marker))
        result["pod_id"] = warm_serve["instance_id"]
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
    write_marker(out_dir, marker)
    return {
        "status": marker["status"],
        "pod_id": pod_id,
        "teardown": teardown,
        "teardown_proof": proof,
        "pending_teardown_close": pending_close,
        "paid_provider_lane_lease": lease_set.summary,
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
