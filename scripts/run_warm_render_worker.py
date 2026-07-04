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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for candidate in (str(SRC_DIR), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from blueprint_pipeline.isaac_g1_kitchen_parity_job import (  # noqa: E402
    JOB_MANIFEST_FILENAME,
    run_isaac_g1_kitchen_parity_job,
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
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


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
        marker_timeout=marker_timeout,
        warm_candidates=tuple(warm_candidates or ()),
        serve=True,
        serve_idle_timeout_s=serve_idle_timeout_s,
        serve_max_jobs=serve_max_jobs,
        serve_ready_timeout=serve_ready_timeout,
    )
    manifest_path = out_path / JOB_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
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
            "manifest_path": str(manifest_path),
            "inbox_put_url_file": warm_serve.get("inbox_put_url_file"),
            "output_get_url_file": warm_serve.get("output_get_url_file"),
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
    marker = read_marker(out_dir)
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
    return submit_fn(
        manifest_path=marker["manifest_path"],
        scenarios_path=scenarios_path,
        out_dir=Path(out_dir) / "warm_results",
        timeout_s=timeout_s,
        interval_s=interval_s,
        stop_after=stop_after,
    )


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
    teardown = provider.terminate(pod_id)
    marker["status"] = "terminated"
    marker["terminated_at"] = _utc_now_iso()
    marker["teardown"] = teardown
    write_marker(out_dir, marker)
    return {"status": "terminated", "pod_id": pod_id, "teardown": teardown}


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
    start.add_argument("--warm-candidate", action="append", default=[])
    start.add_argument("--marker-timeout", type=int, default=900)
    start.add_argument("--serve-idle-timeout", type=float, default=1800.0,
                       help="pod exits by itself after this many idle seconds (cost ceiling)")
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
