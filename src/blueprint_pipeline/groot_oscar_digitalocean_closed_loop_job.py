"""DigitalOcean launcher for the sealed GR00T+OSCAR closed-loop image.

This is the paid execution path for the pushed ``blueprint-groot-oscar-eval``
image. It deliberately reuses the existing provider-neutral object-store
transport and teardown watcher: the worker uploads ``bootstrap.json`` progress
plus a final ``isaac_runtime_result.json`` zip to a signed PUT URL, the host
polls the signed GET URL, and the DigitalOcean droplet is destroyed with
provider-API verification.

Claim boundary: this launcher proves provider startup and closed-loop artifact
production only when the returned manifest says so. Generated-video success,
forward/inverse consistency, real-world task success, and physical robot
readiness remain separate gates.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json
from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .groot_oscar_closed_loop_image import (
    IMAGE_REF_ENV,
    SEALED_CONFIRMED_ENV,
    build_sealed_launch_plan,
    sealed_image_contract,
)
from .isaac_particlefield_render_job import stage_bundle, watch_and_collect
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    open_pending_teardown,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)

SCHEMA_VERSION = "groot_oscar_digitalocean_closed_loop_job.v1"
LANE = "kitchen_g1_groot_oscar_closed_loop"
JOB_MANIFEST_FILENAME = "groot_oscar_digitalocean_closed_loop_job_manifest.json"
DEFAULT_PROVIDER = "digitalocean"
DEFAULT_MAX_HOURLY_RATE_USD = 3.5
DEFAULT_CONTAINER_DISK_GB = 220
DEFAULT_VOLUME_GB = 120
DEFAULT_MAX_SECONDS = 7200
DEFAULT_EPISODE_MAX_STEPS = 48
WORKER_PROGRESS_STALL_PHASES = (
    "container_bash_started",
    "inputs_ready",
    "healthcheck_passed",
    "groot_server_ready",
)


def _string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64_text(text: str) -> str:
    return _b64_bytes(text.encode("utf-8"))


def _json_b64(payload: Mapping[str, Any]) -> str:
    return _b64_text(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")))


def _write_input_bundle(
    *,
    bundle_zip: Path,
    plan: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    seed_path: Path,
    task_prompt: str,
) -> Path:
    ensure_dir(bundle_zip.parent)
    manifest = {
        "schema_version": "groot_oscar_closed_loop_input_bundle.v1",
        "generated_at": utc_now_iso(),
        "seed_filename": "initial_policy_frame.png",
        "route_filename": "route.json",
        "task_prompt": task_prompt,
        "sealed_launch_plan": dict(plan),
        "claim_boundary": (
            "Input bundle stages the requested closed-loop prompt, route, and "
            "seed frame. It is not WAM generation, manipulation success, "
            "forward/inverse consistency, or physical robot proof."
        ),
    }
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(seed_path, "initial_policy_frame.png")
        zf.writestr("route.json", json.dumps(dict(route_payload), indent=2))
        zf.writestr("task_prompt.txt", task_prompt)
        zf.writestr("sealed_launch_plan.json", json.dumps(dict(plan), indent=2))
        zf.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    return bundle_zip


def _write_job_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    write_json(out_dir / JOB_MANIFEST_FILENAME, payload)
    return payload


def _read_json_mapping(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _shell_join(argv: Sequence[Any]) -> str:
    return shlex.join([str(item) for item in argv])


def build_worker_bootstrap_script(plan: Mapping[str, Any]) -> str:
    """Container-side script run inside the sealed image.

    The DigitalOcean cloud-init wrapper starts this script as the container's
    bash command. Inputs ride through environment variables so the host does not
    need SSH, and progress/results are uploaded as a zip to the signed PUT URL
    used by the existing watcher.
    """
    env_exports = "\n".join(
        f"export {shlex.quote(str(key))}={shlex.quote(str(value))}"
        for key, value in sorted(_mapping(plan.get("env")).items())
    )
    groot_cmd = _shell_join(plan.get("groot_server_command") or [])
    closed_loop_cmd = _shell_join(plan.get("closed_loop_command") or [])
    return f"""set -euo pipefail
mkdir -p /workspace/closed_loop_out /workspace/out
{env_exports}
cat > /workspace/upload_progress.py <<'PY'
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

workspace = Path("/workspace")
phase = os.environ.get("BLUEPRINT_BOOTSTRAP_PHASE", "unknown")
bootstrap = {{
    "schema_version": "groot_oscar_closed_loop_bootstrap.v1",
    "phase": phase,
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "launch_session_id": os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID"),
    "raw_secret_values_recorded": False,
}}
(workspace / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
zip_path = workspace / "out" / "groot_oscar_closed_loop_worker_output.zip"
include = [
    workspace / "bootstrap.json",
    workspace / "sealed_launch_plan.json",
    workspace / "seed_provenance.json",
    workspace / "initial_policy_frame.png",
    workspace / "route.json",
    workspace / "task_prompt.txt",
    workspace / "groot_oscar_image_healthcheck.json",
    workspace / "groot_oscar_image_healthcheck.stderr.log",
    workspace / "groot_server.log",
    workspace / "closed_loop_stdout.log",
    workspace / "closed_loop_stderr.log",
    workspace / "isaac_runtime_result.json",
]
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in include:
        if path.is_file():
            zf.write(path, path.relative_to(workspace).as_posix())
    out_dir = workspace / "closed_loop_out"
    if out_dir.is_dir():
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(workspace).as_posix())
put_url = os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL", "")
if put_url:
    subprocess.run(["curl", "-fsS", "-X", "PUT", "--upload-file", str(zip_path), put_url], check=False)
PY

cat > /workspace/write_result.py <<'PY'
import json
import os
from pathlib import Path

rc = int(os.environ.get("BLUEPRINT_CLOSED_LOOP_RC", "0"))
failure = os.environ.get("BLUEPRINT_WORKER_FAILURE", "").strip()
manifest_path = Path("/workspace/closed_loop_out/oscar_isaac_closed_loop_manifest.json")
manifest = {{}}
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {{"status": "blocked", "blockers": [f"closed_loop_manifest_unreadable:{{type(exc).__name__}}"]}}
blockers = list(manifest.get("blockers") or [])
if failure:
    blockers.append(failure)
if rc != 0:
    blockers.append(f"closed_loop_command_exit_{{rc}}")
status = "completed" if rc == 0 and manifest.get("status") == "completed" and not blockers else "blocked"
result = {{
    "schema_version": "groot_oscar_closed_loop_worker_result.v1",
    "status": status,
    "blockers": blockers,
    "closed_loop_return_code": rc,
    "closed_loop_manifest_path": str(manifest_path),
    "closed_loop_manifest": manifest,
    "raw_secret_values_recorded": False,
    "claim_boundary": {{
        "worker_completed_is_not_generated_video_success": True,
        "worker_completed_is_not_forward_inverse_consistency": True,
        "worker_completed_is_not_real_world_task_success": True,
        "worker_completed_is_not_physical_robot_readiness": True,
    }},
}}
Path("/workspace/isaac_runtime_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
PY

upload_phase() {{
  BLUEPRINT_BOOTSTRAP_PHASE="$1" python /workspace/upload_progress.py || true
}}

upload_phase container_bash_started

python - <<'PY'
import base64
import os
from pathlib import Path

Path("/workspace/initial_policy_frame.png").write_bytes(
    base64.b64decode(os.environ["BLUEPRINT_INITIAL_POLICY_FRAME_B64"])
)
Path("/workspace/route.json").write_text(
    base64.b64decode(os.environ["BLUEPRINT_ROUTE_JSON_B64"]).decode("utf-8"),
    encoding="utf-8",
)
Path("/workspace/task_prompt.txt").write_text(
    os.environ["BLUEPRINT_TASK_PROMPT"],
    encoding="utf-8",
)
Path("/workspace/sealed_launch_plan.json").write_text(
    base64.b64decode(os.environ["BLUEPRINT_SEALED_LAUNCH_PLAN_B64"]).decode("utf-8"),
    encoding="utf-8",
)
Path("/workspace/seed_provenance.json").write_text(
    base64.b64decode(os.environ.get("BLUEPRINT_SEED_PROVENANCE_B64", "e30=")).decode("utf-8"),
    encoding="utf-8",
)
PY
upload_phase inputs_ready

set +e
python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda \
  > /workspace/groot_oscar_image_healthcheck.json \
  2> /workspace/groot_oscar_image_healthcheck.stderr.log
HEALTHCHECK_RC=$?
set -e
if [ "$HEALTHCHECK_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$HEALTHCHECK_RC" \
    BLUEPRINT_WORKER_FAILURE="groot_oscar_image_healthcheck_failed" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$HEALTHCHECK_RC"
fi
upload_phase healthcheck_passed

{groot_cmd} > /workspace/groot_server.log 2>&1 &
GROOT_PID=$!
cleanup() {{
  kill "$GROOT_PID" >/dev/null 2>&1 || true
}}
trap cleanup EXIT

set +e
python - <<'PY'
import socket
import time

deadline = time.time() + 900
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", 5550))
    except OSError:
        time.sleep(5)
    else:
        sock.close()
        break
else:
    raise SystemExit("groot_policy_server_not_ready")
PY
GROOT_READY_RC=$?
set -e
if [ "$GROOT_READY_RC" -ne 0 ]; then
  BLUEPRINT_CLOSED_LOOP_RC="$GROOT_READY_RC" \
    BLUEPRINT_WORKER_FAILURE="groot_policy_server_not_ready" \
    python /workspace/write_result.py
  upload_phase runner_done
  exit "$GROOT_READY_RC"
fi
upload_phase groot_server_ready

set +e
{closed_loop_cmd} > /workspace/closed_loop_stdout.log 2> /workspace/closed_loop_stderr.log
RC=$?
set -e

BLUEPRINT_CLOSED_LOOP_RC="$RC" python /workspace/write_result.py
upload_phase runner_done
exit "$RC"
"""


def build_launch_spec(
    *,
    job_dir: Path,
    image_ref: str,
    start_frame: Path,
    route_payload: Mapping[str, Any],
    task_prompt: str,
    plan: Mapping[str, Any],
    seed_provenance: Mapping[str, Any] | None = None,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
) -> RenderLaunchSpec:
    put_url = (job_dir / "provider_output_put_url.txt").read_text(encoding="utf-8").strip()
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text(encoding="utf-8").strip()
    env = {
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "CUDA_VISIBLE_DEVICES": "0",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "BLUEPRINT_INITIAL_POLICY_FRAME_B64": _b64_bytes(start_frame.read_bytes()),
        "BLUEPRINT_ROUTE_JSON_B64": _b64_text(json.dumps(dict(route_payload), indent=2)),
        "BLUEPRINT_TASK_PROMPT": task_prompt,
        "BLUEPRINT_SEALED_LAUNCH_PLAN_B64": _json_b64(plan),
        "BLUEPRINT_SEED_PROVENANCE_B64": _json_b64(seed_provenance or {}),
        SEALED_CONFIRMED_ENV: "true",
    }
    return RenderLaunchSpec(
        name="blueprint-groot-oscar-closed-loop",
        image=image_ref,
        env=env,
        bootstrap_argv=["-lc", build_worker_bootstrap_script(plan)],
        entrypoint=["bash"],
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        max_hourly_rate_usd=max_hourly_rate_usd,
        min_gpu_ram_mb=48000,
    )


def _prelaunch_spend_guard(
    *,
    allow_paid: bool,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: float,
) -> dict[str, Any]:
    seconds = max(0, int(max_seconds or 0))
    hourly = float(max_hourly_rate_usd)
    estimated = round(hourly * (seconds / 3600.0), 4)
    blockers: list[str] = []
    if not allow_paid:
        blockers.append("paid_launch_not_requested")
    if max_spend_usd is None:
        blockers.append("groot_oscar_closed_loop_max_spend_usd_missing")
    elif float(max_spend_usd) <= 0:
        blockers.append("groot_oscar_closed_loop_max_spend_usd_must_be_positive")
    elif estimated > float(max_spend_usd):
        blockers.append("groot_oscar_closed_loop_estimated_spend_exceeds_budget")
    can_launch = bool(allow_paid and not blockers)
    return {
        "schema_version": "groot_oscar_closed_loop_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "provider": DEFAULT_PROVIDER,
        "allow_paid": bool(allow_paid),
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": max_spend_usd,
        "estimated_max_spend_usd": estimated,
        "max_hourly_rate_usd": hourly,
        "max_seconds": seconds,
        "blockers": blockers,
        "claim_boundary": {
            "spend_guard_only": True,
            "can_launch_is_not_provider_success": True,
            "can_launch_is_not_task_success": True,
            "no_provider_api_call_before_can_launch": True,
        },
    }


def _teardown_proof_from_watch(*, instance_id: str, watch: Mapping[str, Any]) -> dict[str, Any]:
    teardown = _mapping(watch.get("teardown"))
    reason = _string(watch.get("teardown_reason")).lower()
    status = _string(teardown.get("status")).lower()
    if status in {"stopped", "preserved", "skipped"} or reason in {
        "left_running_by_request",
        "runner_done_preserved_for_warm_reuse",
    }:
        return build_teardown_proof(
            provider=DEFAULT_PROVIDER,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=reason or status or "kept_alive",
        )
    verification = _mapping(teardown.get("verification"))
    observed_status = _string(verification.get("provider_status")).lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=DEFAULT_PROVIDER,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=DEFAULT_PROVIDER,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=utc_now_iso() if status == "terminated" else None,
    )


def run_groot_oscar_digitalocean_closed_loop_job(
    *,
    start_frame: str | Path,
    route_file: str | Path,
    task_prompt: str,
    out_dir: str | Path,
    steps: int = DEFAULT_EPISODE_MAX_STEPS,
    oscar_height: int = 480,
    oscar_width: int = 640,
    allow_paid: bool = False,
    max_spend_usd: float | None = None,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    max_hourly_rate_usd: float = DEFAULT_MAX_HOURLY_RATE_USD,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    volume_gb: int = DEFAULT_VOLUME_GB,
    seed_provenance: Mapping[str, Any] | None = None,
    key_prefix: str = "blueprint/groot-oscar-closed-loop",
    image_ref: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    ensure_dir(out)
    seed = Path(start_frame)
    route_path = Path(route_file)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "provider": DEFAULT_PROVIDER,
        "start_frame": str(seed),
        "route_file": str(route_path),
        "steps": int(steps),
        "oscar_height": int(oscar_height),
        "oscar_width": int(oscar_width),
        "episode_termination": "task_completion_or_step_cap",
        "container_disk_gb": int(container_disk_gb),
        "volume_gb": int(volume_gb),
        "raw_secret_values_recorded": False,
    }
    if not seed.is_file():
        manifest["blockers"].append("start_frame_missing")
        return _write_job_manifest(out, manifest)
    if not route_path.is_file():
        manifest["blockers"].append("route_file_missing")
        return _write_job_manifest(out, manifest)
    route_payload = json.loads(route_path.read_text(encoding="utf-8"))
    if not isinstance(route_payload, Mapping):
        manifest["blockers"].append("route_file_must_contain_json_object")
        return _write_job_manifest(out, manifest)
    env = {SEALED_CONFIRMED_ENV: "true"}
    if image_ref:
        env[IMAGE_REF_ENV] = image_ref
    contract = sealed_image_contract(env=env)
    manifest["sealed_image_contract"] = contract
    if contract.get("sealed_active") is not True:
        manifest["blockers"].extend(contract.get("blockers") or ["sealed_image_not_active"])
        return _write_job_manifest(out, manifest)
    plan = build_sealed_launch_plan(
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=int(steps),
        task_prompt=task_prompt,
        output_dir="/workspace/closed_loop_out",
        oscar_height=int(oscar_height),
        oscar_width=int(oscar_width),
        env=env,
    )
    manifest["sealed_launch_plan"] = plan
    if plan.get("sealed_active") is not True or plan.get("blockers"):
        manifest["blockers"].extend(plan.get("blockers") or ["sealed_launch_plan_blocked"])
        return _write_job_manifest(out, manifest)
    provider = get_render_provider(DEFAULT_PROVIDER)
    manifest["provider_available"] = provider.available()
    prelaunch = _prelaunch_spend_guard(
        allow_paid=allow_paid,
        max_spend_usd=max_spend_usd,
        max_seconds=max_seconds,
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    manifest["prelaunch_spend_guard"] = prelaunch
    if not allow_paid:
        bundle_zip = _write_input_bundle(
            bundle_zip=out / "groot_oscar_closed_loop_input_bundle.zip",
            plan=plan,
            route_payload=route_payload,
            seed_path=seed,
            task_prompt=task_prompt,
        )
        manifest["bundle_zip"] = str(bundle_zip)
        manifest["status"] = "prepared"
        manifest["note"] = "local bundle and sealed launch plan prepared; re-run with --allow-paid to launch DigitalOcean"
        return _write_job_manifest(out, manifest)
    if prelaunch.get("can_launch") is not True:
        manifest["blockers"].append("groot_oscar_closed_loop_prelaunch_spend_guard_not_passed")
        manifest["blockers"].extend(prelaunch.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    capacity = provider.capacity_preflight()
    manifest["provider_capacity_preflight"] = capacity
    if capacity.get("status") == "blocked":
        manifest["blockers"].extend(capacity.get("blockers") or [])
        manifest["blockers"].append("provider_capacity_unavailable_before_staging")
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return _write_job_manifest(out, manifest)
    bundle_zip = _write_input_bundle(
        bundle_zip=out / "groot_oscar_closed_loop_input_bundle.zip",
        plan=plan,
        route_payload=route_payload,
        seed_path=seed,
        task_prompt=task_prompt,
    )
    manifest["bundle_zip"] = str(bundle_zip)
    job_dir = out / "object_store_real_run"
    staged = stage_bundle(bundle_zip, job_dir, key_prefix=key_prefix)
    manifest["staging"] = {"status": staged.get("status")}
    if staged.get("status") != "completed":
        manifest["blockers"].append("staging_failed")
        manifest["staging"]["stderr_tail"] = staged.get("stderr_tail")
        return _write_job_manifest(out, manifest)
    spec = build_launch_spec(
        job_dir=job_dir,
        image_ref=str(contract["image_ref"]),
        start_frame=seed,
        route_payload=route_payload,
        task_prompt=task_prompt,
        plan=plan,
        seed_provenance=seed_provenance,
        container_disk_gb=int(container_disk_gb),
        volume_gb=int(volume_gb),
        max_hourly_rate_usd=max_hourly_rate_usd,
    )
    request = provider.build_request(spec, job_dir)
    request["prelaunch_spend_guard"] = prelaunch
    manifest["launch_request_shape"] = {
        "provider": DEFAULT_PROVIDER,
        "image": spec.image,
        "container_disk_gb": spec.container_disk_gb,
        "volume_gb": spec.volume_gb,
        "max_hourly_rate_usd": spec.max_hourly_rate_usd,
        "min_gpu_ram_mb": spec.min_gpu_ram_mb,
        "has_start_frame_payload": bool(spec.env.get("BLUEPRINT_INITIAL_POLICY_FRAME_B64")),
        "has_route_payload": bool(spec.env.get("BLUEPRINT_ROUTE_JSON_B64")),
    }
    pending = open_pending_teardown(
        provider=DEFAULT_PROVIDER,
        lane=LANE,
        run_id=f"groot-oscar-do-{uuid.uuid4().hex}",
        job_dir=str(job_dir),
        max_age_seconds=int(max_seconds) + 1800,
    )
    manifest["pending_teardown_record"] = pending["path"]
    launch = provider.launch(job_dir, request, cold=True)
    manifest["launch"] = launch
    if launch.get("instance_id"):
        bind_pending_teardown_instance(pending["path"], str(launch["instance_id"]))
    if launch.get("status") != "launched":
        if not launch.get("instance_id"):
            cancel_pending_teardown(
                pending["path"],
                reason="launch_returned_no_allocation",
                evidence=launch,
            )
        manifest["blockers"].append("launch_failed")
        return _write_job_manifest(out, manifest)
    watch = watch_and_collect(
        job_dir,
        out / "closed_loop_output",
        str(launch["instance_id"]),
        provider=provider,
        max_seconds=int(max_seconds),
        progress_timeout_seconds=900,
        progress_stall_phases=WORKER_PROGRESS_STALL_PHASES,
    )
    manifest["closed_loop_run"] = watch
    teardown_proof = _teardown_proof_from_watch(
        instance_id=str(launch["instance_id"]),
        watch=watch,
    )
    manifest["teardown_proof"] = teardown_proof
    manifest["pending_teardown_close"] = close_pending_teardown(pending["path"], teardown_proof)
    manifest["status"] = "completed" if watch.get("status") == "completed" else "blocked"
    if manifest["status"] != "completed":
        manifest["blockers"].append("closed_loop_run_not_completed")
    return _write_job_manifest(out, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-frame", required=True)
    parser.add_argument("--route-file", required=True)
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_EPISODE_MAX_STEPS)
    parser.add_argument("--oscar-height", type=int, default=480)
    parser.add_argument("--oscar-width", type=int, default=640)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-spend-usd", type=float, default=None)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-hourly-rate-usd", type=float, default=DEFAULT_MAX_HOURLY_RATE_USD)
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    parser.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    parser.add_argument("--seed-provenance-file", default=None)
    parser.add_argument("--key-prefix", default="blueprint/groot-oscar-closed-loop")
    parser.add_argument("--image-ref", default=None)
    args = parser.parse_args(argv)
    result = run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=args.start_frame,
        route_file=args.route_file,
        task_prompt=args.task_prompt,
        out_dir=args.out_dir,
        steps=args.steps,
        oscar_height=args.oscar_height,
        oscar_width=args.oscar_width,
        allow_paid=args.allow_paid,
        max_spend_usd=args.max_spend_usd,
        max_seconds=args.max_seconds,
        max_hourly_rate_usd=args.max_hourly_rate_usd,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        seed_provenance=_read_json_mapping(args.seed_provenance_file),
        key_prefix=args.key_prefix,
        image_ref=args.image_ref,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in {"completed", "prepared"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
