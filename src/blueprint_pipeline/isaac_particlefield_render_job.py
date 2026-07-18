"""End-to-end Isaac Sim ParticleField splat render job (productionized).

Turns a captured Gaussian splat into Isaac-RTX-rendered frames of the real scene, as a
repeatable pipeline step (not ad-hoc scripts). The chain:

  capture splat (.ply/.spz/.usdc)
    -> standard 3DGS PLY            (gaussian_splat_decode)
    -> ParticleField3DGaussianSplat USD   (particlefield_usd, pure pxr)
    -> 6 free eval cameras          (splat_scene_analysis)
    -> bundle + stage to object store    (wam_provider_object_store, DO Spaces signed URLs)
    -> RunPod base-Isaac pod runs scripts/run_isaac_splat_nurec_render.py via a bootstrap
       (warm-host restart first, else cold on-demand) -> RTX frames + MP4 uploaded
    -> poll, collect, teardown.

Paid GPU launches are gated behind ``allow_paid=True``; otherwise the job stages everything
and returns a launchable plan (``status='prepared'``) without spending. Secrets are
file-based under ``~/.blueprint-secrets`` and never logged. Truth boundary: Isaac RTX
render evidence of the captured scene only.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso
from .gpu_render_providers import RenderLaunchSpec, get_render_provider, list_render_providers
from .paid_lane_guard import (
    PreSpendPreflightBlocked,
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    image_contract_from_ref,
    open_pending_teardown,
    provider_state_from_inspect,
    require_pre_spend_preflight,
)
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_provider_reliability_manifest,
    build_teardown_proof,
)

SCHEMA_VERSION = "isaac_particlefield_render_job.v1"
RENDER_LANE = "isaac_particlefield_render"
RELIABILITY_MANIFEST_NAME = "provider_reliability_manifest.json"
POST_MARKER_NO_PROGRESS_TIMEOUT_ENV = "BLUEPRINT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS"
RENDER_POD_HARD_TTL_ENV = "BLUEPRINT_RENDER_POD_HARD_TTL_SECONDS"
RENDER_POD_IDLE_TTL_ENV = "BLUEPRINT_RENDER_POD_IDLE_TTL_SECONDS"
# A booted worker that stops changing bootstrap phase for this long is a stall,
# not patience — it gets terminated instead of billing until max_seconds.
DEFAULT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS = 900
# Independent pod-side backstops. The host watch loop and spend guard should still
# terminate pods first; these limits keep the worker command from running forever if
# the launching/control-plane process dies.
DEFAULT_RENDER_POD_HARD_TTL_SECONDS = 7200
DEFAULT_RENDER_POD_IDLE_TTL_SECONDS = 1800
SECRETS = Path.home() / ".blueprint-secrets"
DEFAULT_WARM_CANDIDATES = (
    "pwbu7wxsvxpr0x", "9zxerj0nm3ow76", "qzgtsh4t27hi7f", "v4bd9u2qhwivb8",
    "y3n5n7t6wvaawe", "1gx9uri0mkrxg9", "ajxbj2ysyow3n9", "usjvua1bwlwhyj",
)
_SPLAT_SUFFIXES = {".ply", ".spz", ".usdc", ".usd", ".usda", ".usdz"}
DEFAULT_POST_MARKER_STALL_PHASES = (
    "container_bash_started",
    "bootstrap_fetching",
    "bootstrap_fetch_connected",
    "bootstrap_fetch_progress",
    "bootstrap_fetch_fetched",
    "bootstrap_extracted",
    "kitchen_fetching",
    "kitchen_fetch_connected",
    "kitchen_fetch_progress",
    "kitchen_fetch_fetched",
    "kitchen_extracting",
    "kitchen_extracted",
)

# Diagnostics-streaming pod bootstrap: fetch bundle -> heartbeat-upload output every 25s ->
# run the Isaac runner -> final upload. Runs under Isaac's python (guaranteed on the image).
BOOTSTRAP = r'''
import os, sys, io, time, json, zipfile, threading, subprocess, urllib.request, pathlib, shutil
OUT="/workspace/out"; BUNDLE="/workspace/bundle"
for d in (OUT, BUNDLE): pathlib.Path(d).mkdir(parents=True, exist_ok=True)
for p in pathlib.Path(OUT).iterdir():
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except Exception: pass
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
GETB=os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI","")
CAMS=os.environ.get("CAMERAS_FILE","cameras.json")
W=os.environ.get("RENDER_WIDTH","1280"); H=os.environ.get("RENDER_HEIGHT","960")
SUB=os.environ.get("RENDER_SUBFRAMES","8"); RTS=os.environ.get("RENDER_RT_SUBFRAMES","64")
WARM=os.environ.get("RENDER_WARMUP_FRAMES","30")
IDLE_TTL=int(os.environ.get("BLUEPRINT_RENDER_POD_IDLE_TTL_SECONDS","1800") or 0)
def putout():
    try:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in pathlib.Path(OUT).rglob("*"):
                if p.is_file():
                    try: z.write(p, p.relative_to(OUT).as_posix())
                    except Exception: pass
        req=urllib.request.Request(PUT, data=buf.getvalue(), method="PUT", headers={"Content-Type":"application/zip"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception: pass
def mark(ph, **k):
    try: json.dump({"phase":ph, **k}, open(OUT+"/bootstrap.json","w"))
    except Exception: pass
    putout()
def hb():
    while True:
        time.sleep(25); putout()
mark("bootstrap_fetching")
data=urllib.request.urlopen(GETB, timeout=600).read()
zipfile.ZipFile(io.BytesIO(data)).extractall(BUNDLE)
mark("bootstrap_extracted", files=sorted(os.listdir(BUNDLE)))
threading.Thread(target=hb, daemon=True).start()
cmd=["/isaac-sim/python.sh", BUNDLE+"/run_isaac_splat_nurec_render.py",
     "--usdc", BUNDLE+"/scene_particlefield.usdc", "--cameras", BUNDLE+"/"+CAMS,
     "--out-dir", OUT, "--width", W, "--height", H,
     "--warmup-frames", WARM, "--subframes", SUB, "--rt-subframes", RTS]
mark("runner_starting", cmd=cmd)
rc=subprocess.call(cmd)
mark("runner_done", rc=rc, idle_ttl_seconds=IDLE_TTL)
# Keep the container process alive after runner completion so RunPod does not restart it and
# clobber the final output object before the parent collector observes runner_done.
idle_deadline = time.time()+IDLE_TTL if IDLE_TTL > 0 else None
while True:
    if idle_deadline is not None and time.time() >= idle_deadline:
        mark("pod_idle_ttl_exceeded", idle_ttl_seconds=IDLE_TTL)
        sys.exit(rc if rc else 0)
    time.sleep(30); putout()
'''


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_secret(name: str) -> str | None:
    p = SECRETS / name
    return p.read_text().strip() if p.is_file() else None


def default_image() -> str:
    ref = _read_secret("isaac_eval_worker_image_ref")
    return ref or "docker.io/nijelhunt/blueprint-isaac-eval-worker:20260626-faststart-amd64"


# bash-level early marker (system python3) — proves the container+bash ran even if Isaac's
# python fails to boot, so "container never started" is distinguishable from "Isaac failed".
_EARLY_MARKER = r'''
import os, io, json, zipfile, pathlib, urllib.request
OUT="/workspace/out"; pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
json.dump({"phase":"container_bash_started"}, open(OUT+"/bootstrap.json","w"))
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
buf=io.BytesIO()
with zipfile.ZipFile(buf,"w") as z: z.write(OUT+"/bootstrap.json","bootstrap.json")
try: urllib.request.urlopen(urllib.request.Request(PUT,data=buf.getvalue(),method="PUT",headers={"Content-Type":"application/zip"}),timeout=60).read()
except Exception: pass
'''


def docker_start_cmd() -> list[str]:
    # write both scripts to files (robust) — avoid stdin-to-python tricks that silently no-op.
    script = (
        "set +e\n"
        "mkdir -p /workspace/out\n"
        f"export {RENDER_POD_HARD_TTL_ENV}=${{{RENDER_POD_HARD_TTL_ENV}:-{DEFAULT_RENDER_POD_HARD_TTL_SECONDS}}}\n"
        f"export {RENDER_POD_IDLE_TTL_ENV}=${{{RENDER_POD_IDLE_TTL_ENV}:-{DEFAULT_RENDER_POD_IDLE_TTL_SECONDS}}}\n"
        f"if [ \"${{{RENDER_POD_HARD_TTL_ENV}}}\" -gt 0 ] 2>/dev/null; then\n"
        "  (\n"
        f"    sleep \"${{{RENDER_POD_HARD_TTL_ENV}}}\"\n"
        "    printf '%s\\n' '{\"phase\":\"pod_hard_ttl_exceeded\",\"source\":\"bash_watchdog\"}' > /workspace/out/bootstrap.json\n"
        "    pkill -TERM -P $$ 2>/dev/null || true\n"
        "    sleep 10\n"
        "    pkill -KILL -P $$ 2>/dev/null || true\n"
        "    kill -TERM $$ 2>/dev/null || true\n"
        "  ) &\n"
        "  BLUEPRINT_RENDER_POD_WATCHDOG_PID=$!\n"
        "fi\n"
        "cat > /workspace/early.py <<'EARLYEOF'\n" + _EARLY_MARKER + "\nEARLYEOF\n"
        "(python3 /workspace/early.py 2>/dev/null || /isaac-sim/python.sh /workspace/early.py 2>/dev/null) || true\n"
        "cat > /workspace/boot.py <<'PYEOF'\n" + BOOTSTRAP + "\nPYEOF\n"
        "/isaac-sim/python.sh /workspace/boot.py 2>&1 | tee /workspace/out/runner_console.log\n"
    )
    return ["-lc", script]


# ----------------------------- asset + bundle prep -----------------------------

def ensure_particlefield_usd(source: str | Path, out_dir: Path) -> dict:
    """Resolve the render asset to a ParticleField USD. Accepts a ParticleField .usdc/.usd
    directly, a standard 3DGS PLY (authored), or a compressed PLY/SPZ (decoded then authored)."""
    source = Path(source)
    if source.suffix.lower() in {".usdc", ".usd", ".usda"}:
        return {"status": "completed", "usdc": str(source), "source_kind": "particlefield_usd"}
    from .gaussian_splat_decode import convert_to_standard_ply, read_standard_3dgs_ply
    from .particlefield_usd import write_particlefield_usd

    out_dir.mkdir(parents=True, exist_ok=True)
    std = out_dir / "scene_standard.ply"
    # try reading as a standard 3DGS PLY; else decode (compressed PlayCanvas / SPZ)
    try:
        read_standard_3dgs_ply(source)
        std = source
    except Exception:  # noqa: BLE001
        dec = convert_to_standard_ply(source, std)
        if dec.get("status") != "completed":
            return {"status": "blocked", "blockers": dec.get("blockers", ["decode_failed"]), "decode": dec}
    usdc = out_dir / "scene_particlefield.usdc"
    auth = write_particlefield_usd(std, usdc)
    if auth.get("status") != "completed":
        return {"status": "blocked", "blockers": auth.get("blockers", ["authoring_failed"]), "authoring": auth}
    return {"status": "completed", "usdc": str(usdc), "standard_ply": str(std),
            "splat_count": auth.get("splat_count"), "source_kind": "authored"}


def derive_cameras_for(standard_ply: str | Path, *, up_axis: int | None = None) -> list[dict]:
    from .gaussian_splat_decode import read_standard_3dgs_ply
    from .splat_scene_analysis import analyze_scene, derive_eval_cameras

    geom = analyze_scene(read_standard_3dgs_ply(standard_ply), up_axis=up_axis)
    return derive_eval_cameras(geom)


def author_robot_into_stage(
    usdc_path: str | Path,
    *,
    robot_usd: str,
    robot_pose: Sequence[float],
    prim_path: str = "/World/RobotVisual",
    lights_path: str = "/World/CompositeLights",
) -> dict:
    """Author the robot reference + composite lights INTO the stage, pre-open.

    Runtime-added instanceable references can fail to sync into Isaac's Fabric
    scene delegate (the robot composes in USD — bounds, mesh points — but never
    reaches the RTX render index; depth annotators see nothing). Content present
    at ``open_stage`` time always ingests, so the reference is authored locally
    into the bundle's USDC. The referenced asset URL resolves on the WORKER at
    open (no local fetch). Lights are authored INVISIBLE — the splat pass is
    self-emissive and must stay unlit; the runner enables them for the
    robot-only pass so the mesh is not black.
    """
    import math as _math

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # type: ignore

    stage = Usd.Stage.Open(str(usdc_path))
    if stage is None:
        return {"status": "blocked", "blockers": ["stage_open_failed"]}
    prim = stage.DefinePrim(Sdf.Path(prim_path), "Xform")
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(str(robot_usd))
    x, y, z, yaw = (float(v) for v in robot_pose)
    matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), _math.degrees(yaw)))
    matrix.SetTranslateOnly(Gf.Vec3d(x, y, z))
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)

    lights_root = UsdGeom.Xform.Define(stage, Sdf.Path(lights_path))
    UsdGeom.Imageable(lights_root.GetPrim()).MakeInvisible()
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{lights_path}/Dome"))
    dome.CreateIntensityAttr(400.0)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path(f"{lights_path}/Distant"))
    distant.CreateIntensityAttr(2500.0)
    stage.GetRootLayer().Save()
    return {
        "status": "completed",
        "robot_prim_path": prim_path,
        "lights_path": lights_path,
        "robot_usd": str(robot_usd),
        "robot_pose": [x, y, z, yaw],
    }


def build_render_bundle(
    *, usdc_path: str | Path, cameras: Sequence[dict], out_dir: Path,
    canary_camera_id: str = "third_person",
    render_options: Mapping[str, Any] | None = None,
) -> Path:
    """Assemble the pod bundle: ParticleField USD + cameras.json + canary cameras + the runner.

    ``render_options`` (e.g. ``{"robot_usd": ..., "robot_pose": [x, y, z, yaw]}``)
    is written as ``render_options.json`` next to cameras.json — bundle-driven so
    it survives warm pod restarts whose container env was set at create time.
    """
    bundle = out_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "scene_particlefield.usdc").write_bytes(Path(usdc_path).read_bytes())
    (bundle / "cameras.json").write_text(json.dumps(list(cameras)), encoding="utf-8")
    if render_options:
        (bundle / "render_options.json").write_text(
            json.dumps(dict(render_options)), encoding="utf-8"
        )
    canary = [c for c in cameras if c.get("id") == canary_camera_id] or list(cameras)[:1]
    (bundle / "cameras_canary.json").write_text(json.dumps(canary), encoding="utf-8")
    runner = _repo_root() / "scripts" / "run_isaac_splat_nurec_render.py"
    (bundle / "run_isaac_splat_nurec_render.py").write_bytes(runner.read_bytes())
    zip_path = out_dir / "isaac_provider_runtime_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(bundle.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(bundle).as_posix())
    return zip_path


def stage_bundle(bundle_zip: Path, job_dir: Path, *, key_prefix: str = "blueprint/isaac-splat",
                 expiration_seconds: int = 43200) -> dict:
    """Stage the bundle to the object store (DO Spaces) and mint signed GET/PUT URLs."""
    cmd = ["python", "-m", "blueprint_pipeline.wam_provider_object_store",
           "--job-dir", str(job_dir), "--bundle-path", str(bundle_zip),
           "--key-prefix", key_prefix, "--expiration-seconds", str(expiration_seconds)]
    # prefer the current interpreter
    import sys
    cmd[0] = sys.executable
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = (proc.stdout or "").strip()
    manifest = {}
    if out:
        try:
            manifest = json.loads(out[out.index("{"):])
        except Exception:  # noqa: BLE001
            manifest = {}
    ok = manifest.get("status") == "completed" and not manifest.get("blockers")
    return {"status": "completed" if ok else "blocked", "manifest": manifest,
            "stderr_tail": (proc.stderr or "")[-800:] if not ok else ""}


# ----------------------------- provider-agnostic launch spec -----------------------------

def _env_for(
    bundle_url: str,
    put_url: str,
    cameras_file: str,
    *,
    width: int,
    height: int,
    subframes: int,
    rt_subframes: int,
    warmup: int,
    render_pod_hard_ttl_seconds: int = DEFAULT_RENDER_POD_HARD_TTL_SECONDS,
    render_pod_idle_ttl_seconds: int = DEFAULT_RENDER_POD_IDLE_TTL_SECONDS,
) -> dict:
    return {
        "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "CUDA_VISIBLE_DEVICES": "0",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "CAMERAS_FILE": cameras_file,
        "RENDER_WIDTH": str(width), "RENDER_HEIGHT": str(height),
        "RENDER_SUBFRAMES": str(subframes), "RENDER_RT_SUBFRAMES": str(rt_subframes),
        "RENDER_WARMUP_FRAMES": str(warmup),
        RENDER_POD_HARD_TTL_ENV: str(max(0, int(render_pod_hard_ttl_seconds))),
        RENDER_POD_IDLE_TTL_ENV: str(max(0, int(render_pod_idle_ttl_seconds))),
    }


def build_render_launch_spec(
    job_dir: Path, *, image: str, cameras_file: str, width: int = 1280, height: int = 960,
    subframes: int = 8, rt_subframes: int = 64, warmup: int = 30, container_disk_gb: int = 140,
    volume_gb: int = 80, gpu_types: Sequence[str] = ("NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"),
    max_hourly_rate_usd: float = 2.0, min_gpu_ram_mb: int = 24000,
    render_pod_hard_ttl_seconds: int = DEFAULT_RENDER_POD_HARD_TTL_SECONDS,
    render_pod_idle_ttl_seconds: int = DEFAULT_RENDER_POD_IDLE_TTL_SECONDS,
) -> RenderLaunchSpec:
    """Provider-neutral render launch spec (image + env + bootstrap + GPU sizing). The
    env carries the signed bundle GET + output PUT URLs, so any provider just forwards it."""
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text().strip()
    put_url = (job_dir / "provider_output_put_url.txt").read_text().strip()
    env = _env_for(
        bundle_url,
        put_url,
        cameras_file,
        width=width,
        height=height,
        subframes=subframes,
        rt_subframes=rt_subframes,
        warmup=warmup,
        render_pod_hard_ttl_seconds=render_pod_hard_ttl_seconds,
        render_pod_idle_ttl_seconds=render_pod_idle_ttl_seconds,
    )
    return RenderLaunchSpec(
        name="blueprint-isaac-splat-render", image=image, env=env,
        bootstrap_argv=docker_start_cmd(), entrypoint=["bash"],
        container_disk_gb=container_disk_gb, volume_gb=volume_gb,
        gpu_types=tuple(gpu_types), max_hourly_rate_usd=max_hourly_rate_usd,
        min_gpu_ram_mb=min_gpu_ram_mb,
    )


def build_launch_request(job_dir: Path, *, image: str, cameras_file: str, width: int = 1280,
                         height: int = 960, subframes: int = 8, rt_subframes: int = 64,
                         warmup: int = 30, container_disk_gb: int = 140, volume_gb: int = 80,
                         gpu_types: Sequence[str] = ("NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"),
                         provider: str = "runpod") -> dict:
    """The provider-native launch body (no secrets) for the given provider. Defaults to the
    RunPod pod body for backward compatibility; pass ``provider='vast'`` for the Vast shape."""
    spec = build_render_launch_spec(
        job_dir, image=image, cameras_file=cameras_file, width=width, height=height,
        subframes=subframes, rt_subframes=rt_subframes, warmup=warmup,
        container_disk_gb=container_disk_gb, volume_gb=volume_gb, gpu_types=gpu_types)
    return get_render_provider(provider).build_request(spec, job_dir)


def launch_runpod(job_dir: Path, request: dict, *, cold: bool = False,
                  warm_candidates: Sequence[str] = DEFAULT_WARM_CANDIDATES,
                  max_spend_usd: float | None = None) -> dict:
    """Backward-compat RunPod launch (warm-restart-then-cold). Delegates to the provider
    abstraction. Returns {instance_id, pod_id, mode, attempts}; spends money."""
    prelaunch_guard = _particlefield_prelaunch_spend_guard(
        allow_paid=True,
        max_spend_usd=max_spend_usd,
        max_seconds=0,
        max_hourly_rate_usd=request.get("max_hourly_rate_usd"),
    )
    if prelaunch_guard.get("can_launch") is not True:
        return {
            "status": "blocked",
            "reason": "prelaunch_spend_guard_blocked",
            "blockers": [
                "isaac_particlefield_prelaunch_spend_guard_not_passed",
                *(prelaunch_guard.get("blockers") or []),
            ],
            "prelaunch_spend_guard": prelaunch_guard,
        }
    request["prelaunch_spend_guard"] = prelaunch_guard
    provider = get_render_provider("runpod", warm_candidates=warm_candidates)
    result = provider.launch(job_dir, request, cold=cold)
    if result.get("instance_id") and "pod_id" not in result:
        result["pod_id"] = result["instance_id"]
    return result


def stop_pod(pod_id: str) -> dict:
    return get_render_provider("runpod").stop(pod_id)


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _progress_marker_signature(marker: Mapping[str, Any]) -> str:
    """Stable signature for a downloaded bootstrap/progress marker."""
    try:
        return json.dumps(dict(marker), sort_keys=True, default=str, separators=(",", ":"))
    except TypeError:
        return repr(sorted(dict(marker).items()))


def _runner_result_from_dir(out_dir: Path) -> tuple[dict, str | None]:
    for name in ("isaac_runtime_result.json", "isaac_g1_kitchen_parity_result.json"):
        payload = _read_json_file(out_dir / name)
        if payload:
            return payload, name
    return {}, None


def _extract_provider_output_snapshot(data: bytes, parent: Path) -> Path:
    """Extract one provider object into an isolated, validated snapshot dir."""
    snapshot = Path(tempfile.mkdtemp(prefix=".provider-output-snapshot-", dir=parent))
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError(f"unsafe_provider_output_member:{member.filename}")
                target = (snapshot / relative).resolve()
                if os.path.commonpath([str(snapshot.resolve()), str(target)]) != str(
                    snapshot.resolve()
                ):
                    raise ValueError("provider_output_member_escapes_snapshot")
            archive.extractall(snapshot)
        return snapshot
    except BaseException:
        shutil.rmtree(snapshot, ignore_errors=True)
        raise


def _publish_provider_output_snapshot(snapshot: Path, out_dir: Path) -> None:
    """Replace the collected tree exactly; never merge it with an older launch."""
    if out_dir.exists():
        if out_dir.is_symlink() or not out_dir.is_dir():
            out_dir.unlink()
        else:
            shutil.rmtree(out_dir)
    shutil.copytree(snapshot, out_dir)


def watch_and_collect(job_dir: Path, out_dir: Path, instance_id: str, *, provider=None,
                      max_seconds: int = 1200, poll: int = 25,
                      stop_on_success: bool = True,
                      preserve_instance: bool = False,
                      keep_running: bool = False,
                      preserve_blocked_instance: bool = True,
                      progress_timeout_seconds: int = 0,
                      progress_stall_phases: Sequence[str] | None = None) -> dict:
    """Poll the heartbeat-uploaded output (provider-neutral signed GET url), then stop the
    instance via the provider. A worker that reached bootstrap/runner output is preserved with
    stop() even when validation blocks, so it can be warm-restarted. True no-output launch duds are
    terminated. ``provider`` defaults to RunPod for backward compatibility."""
    if provider is None:
        provider = get_render_provider("runpod")
    out_dir.mkdir(parents=True, exist_ok=True)
    get_url = (job_dir / "provider_output_get_url.txt").read_text().strip()
    t0 = time.time()
    last = {}
    last_source = None
    last_boot = {}
    last_console_tail = ""
    done = False
    done_from_final_result_without_runner_done = False
    progress_timeout_seconds = max(0, int(progress_timeout_seconds or 0))
    progress_phase_set = set(progress_stall_phases or DEFAULT_POST_MARKER_STALL_PHASES)
    last_progress_phase = ""
    last_progress_marker_signature = ""
    last_progress_at = t0
    progress_timeout_observed = False
    progress_timeout_detail: dict[str, Any] = {}
    expected_launch_session_id = ""
    nonce_path = job_dir / "launch_session_nonce.txt"
    if nonce_path.is_file():
        try:
            expected_launch_session_id = nonce_path.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            expected_launch_session_id = ""
    stale_launch_snapshot_count = 0
    while time.time() - t0 < max_seconds:
        time.sleep(poll)
        snapshot: Path | None = None
        try:
            data = urllib.request.urlopen(get_url, timeout=60).read()
            snapshot = _extract_provider_output_snapshot(data, out_dir.parent)
            boot = _read_json_file(snapshot / "bootstrap.json")
            if expected_launch_session_id and (
                boot.get("launch_session_id") != expected_launch_session_id
            ):
                stale_launch_snapshot_count += 1
                continue
            _publish_provider_output_snapshot(snapshot, out_dir)
        except Exception:  # noqa: BLE001
            continue
        finally:
            if snapshot is not None:
                shutil.rmtree(snapshot, ignore_errors=True)
        last, last_source = _runner_result_from_dir(out_dir)
        boot = _read_json_file(out_dir / "bootstrap.json")
        if boot:
            last_boot = boot
            phase = str(boot.get("phase") or "")
            marker_signature = _progress_marker_signature(boot)
            if phase and marker_signature != last_progress_marker_signature:
                last_progress_phase = phase
                last_progress_marker_signature = marker_signature
                last_progress_at = time.time()
        console = out_dir / "runner_console.log"
        if console.is_file():
            try:
                last_console_tail = console.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:  # noqa: BLE001
                pass
        if (
            progress_timeout_seconds > 0
            and last_progress_phase in progress_phase_set
            and time.time() - last_progress_at >= progress_timeout_seconds
        ):
            progress_timeout_observed = True
            progress_timeout_detail = {
                "phase": last_progress_phase,
                "timeout_seconds": progress_timeout_seconds,
                "elapsed_since_phase_seconds": round(time.time() - last_progress_at, 1),
                "elapsed_since_progress_seconds": round(time.time() - last_progress_at, 1),
            }
            done = False
            break
        # A launch nonce, when present, was checked before publishing this
        # snapshot. Only then may a terminal phase end the current watch.
        if boot.get("phase") in {"runner_done", "runner_timeout"}:
            done = True
            break
        bootstrap_matches_current_launch = bool(
            expected_launch_session_id
            and boot.get("launch_session_id") == expected_launch_session_id
        )
        if last.get("status") in ("completed", "blocked") and bootstrap_matches_current_launch:
            # The runner writes its final result before closing Isaac. If closing/uploading the final
            # runner_done marker stalls, this current-launch result is still enough to stop polling.
            done = True
            done_from_final_result_without_runner_done = True
            break
        if last.get("status") in ("completed", "blocked") and not boot:
            # Legacy/no-bootstrap fallback. Current bootstraps always write bootstrap.json; when it is
            # present, wait for runner_done so warm-restart stale outputs cannot terminate a fresh run.
            done = True
            break
    if done:
        # one more fetch so the runner's FINAL upload (result json, skeleton trace, last frames)
        # lands before we delete the pod — the heartbeat that flips to runner_done races teardown.
        time.sleep(min(poll, 15))
        snapshot = None
        try:
            data = urllib.request.urlopen(get_url, timeout=60).read()
            snapshot = _extract_provider_output_snapshot(data, out_dir.parent)
            refreshed_boot = _read_json_file(snapshot / "bootstrap.json")
            if expected_launch_session_id and (
                refreshed_boot.get("launch_session_id") != expected_launch_session_id
            ):
                stale_launch_snapshot_count += 1
            else:
                _publish_provider_output_snapshot(snapshot, out_dir)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if snapshot is not None:
                shutil.rmtree(snapshot, ignore_errors=True)
        refreshed, refreshed_source = _runner_result_from_dir(out_dir)
        if refreshed:
            last, last_source = refreshed, refreshed_source
        refreshed_boot = _read_json_file(out_dir / "bootstrap.json")
        if refreshed_boot:
            last_boot = refreshed_boot
            if refreshed_boot.get("phase") == "runner_done":
                done_from_final_result_without_runner_done = False
    completed = bool(done and last.get("status") == "completed")
    runner_started = bool(done or last or last_boot or last_console_tail)
    runner_done_observed = last_boot.get("phase") == "runner_done"
    runner_timeout_observed = last_boot.get("phase") == "runner_timeout"
    timed_out_without_runner_done = not done
    provider_name = str(getattr(provider, "name", "") or "").strip().lower()
    stop_releases_compute = provider_name != "digitalocean"
    provider_snapshot_before_teardown: dict[str, Any] = {}
    if hasattr(provider, "inspect"):
        try:
            snapshot = provider.inspect(instance_id)
            if isinstance(snapshot, Mapping):
                provider_snapshot_before_teardown = dict(snapshot)
        except Exception as exc:  # noqa: BLE001 - teardown remains authoritative
            provider_snapshot_before_teardown = {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    should_preserve_for_warm_reuse = bool(
        done
        and runner_done_observed
        and not done_from_final_result_without_runner_done
        and (completed or preserve_blocked_instance)
        and stop_releases_compute
        and (preserve_instance or (stop_on_success and runner_started))
        and hasattr(provider, "stop")
    )
    if keep_running:
        # Caller explicitly wants the pod left RUNNING (warm follow-up jobs;
        # compute keeps billing until the caller stops it). No teardown call.
        # Unlike preserve_instance (which still terminates duds for cost
        # protection), this is an explicit user directive to keep the pod up.
        teardown = {"status": "skipped", "note": "pod_left_running_by_request"}
        teardown_reason = "left_running_by_request"
    elif should_preserve_for_warm_reuse:
        teardown = provider.stop(instance_id)
        teardown_reason = "runner_done_preserved_for_warm_reuse"
    else:
        # A pod that never reached runner_done is not a reusable warm worker. Terminate it so
        # providers that bill stopped disks (notably RunPod) release storage as well as compute.
        teardown = provider.terminate(instance_id)
        if isinstance(teardown, dict) and hasattr(provider, "inspect"):
            # The DELETE response alone is self-reported; only a follow-up state
            # query can prove the allocation is billing-terminal.
            teardown["verification"] = provider_state_from_inspect(
                provider.inspect(instance_id)
            )
        teardown_reason = (
            "runner_timeout_terminated"
            if runner_timeout_observed
            else
            "final_result_without_runner_done_terminated"
            if done_from_final_result_without_runner_done
            else
            "post_marker_progress_timeout_terminated"
            if progress_timeout_observed
            else
            "timeout_without_runner_done_terminated"
            if timed_out_without_runner_done
            else
            "runner_done_terminated_no_warm_reuse"
            if done and runner_done_observed
            else "launch_dud_terminated"
        )
    return {"status": "completed" if completed else "blocked",
            "runner_result": last, "runner_result_source": last_source,
            "last_bootstrap": last_boot, "runner_console_tail": last_console_tail,
            "teardown": teardown, "teardown_reason": teardown_reason,
            "timed_out_without_runner_done": timed_out_without_runner_done,
            "post_marker_progress_timeout_observed": progress_timeout_observed,
            "post_marker_progress_timeout": progress_timeout_detail,
            "runner_done_observed": runner_done_observed,
            "runner_timeout_observed": runner_timeout_observed,
            "final_result_without_runner_done": done_from_final_result_without_runner_done,
            "stale_launch_snapshot_count": stale_launch_snapshot_count,
            "provider_snapshot_before_teardown": provider_snapshot_before_teardown,
            "elapsed_seconds": round(time.time() - t0, 1)}


def _launch_shape_summary(provider_name: str, request: dict) -> dict:
    """Provider-agnostic one-line summary of the launch body for the plan/manifest."""
    if provider_name == "vast":
        return {"provider": "vast", "image": request.get("image"),
                "disk_gb": request.get("disk"),
                "gpu_selection": "vast_offer_search_rt_capable",
                "max_hourly_rate_usd": request.get("max_hourly_rate_usd")}
    return {"provider": provider_name, "image": request.get("imageName"),
            "gpuTypeIds": request.get("gpuTypeIds"),
            "containerDiskInGb": request.get("containerDiskInGb"),
            "volumeInGb": request.get("volumeInGb")}


def _default_post_marker_timeout() -> int:
    raw = os.getenv(POST_MARKER_NO_PROGRESS_TIMEOUT_ENV, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS
    return max(0, value)


def _phase(passed: bool, blockers: Sequence[str] = (), **fields: Any) -> dict:
    return {
        "status": "PASS" if passed else "FAIL",
        "blockers": [str(b) for b in blockers if str(b).strip()],
        **fields,
    }


def _render_runtime_contract(
    progress_timeout_seconds: int,
    *,
    render_pod_hard_ttl_seconds: int = DEFAULT_RENDER_POD_HARD_TTL_SECONDS,
    render_pod_idle_ttl_seconds: int = DEFAULT_RENDER_POD_IDLE_TTL_SECONDS,
) -> dict[str, Any]:
    timeout = max(0, int(progress_timeout_seconds or 0))
    return {
        "startup_marker": "container_bash_started",
        "progress_marker": "bootstrap.json",
        "startup_timeout_seconds": timeout,
        "no_progress_timeout_seconds": timeout,
        "progress_stall_phases": list(DEFAULT_POST_MARKER_STALL_PHASES),
        "pod_side_watchdog": {
            "hard_ttl_env": RENDER_POD_HARD_TTL_ENV,
            "hard_ttl_seconds": max(0, int(render_pod_hard_ttl_seconds)),
            "idle_ttl_env": RENDER_POD_IDLE_TTL_ENV,
            "idle_ttl_seconds": max(0, int(render_pod_idle_ttl_seconds)),
            "owner": "render_worker_bootstrap",
        },
    }


def _credential_present_from_availability(availability: Mapping[str, Any]) -> bool:
    if availability.get("available") is True:
        return True
    reason = str(availability.get("reason") or "").strip().lower()
    if any(token in reason for token in ("key_missing", "credential_missing", "api_key_missing")):
        return False
    return bool(reason)


def _render_pre_spend_preflight(
    *,
    provider_name: str,
    availability: Mapping[str, Any] | None,
    image_ref: str,
    progress_timeout_seconds: int,
    spend_gate_open: bool,
) -> dict[str, Any]:
    """Route the render lane through the shared pre-spend chokepoint.

    Raises :class:`PreSpendPreflightBlocked` (with the FAIL preflight attached)
    instead of returning a failing contract — no billable call may follow.
    """
    availability_evidence = dict(availability or {})
    if "detail" not in availability_evidence and availability_evidence.get("reason"):
        availability_evidence["detail"] = availability_evidence.get("reason")
    return require_pre_spend_preflight(
        lane=RENDER_LANE,
        provider=provider_name,
        credential_present=_credential_present_from_availability(availability_evidence),
        capacity_evidence=availability_evidence,
        image_contract=image_contract_from_ref(image_ref),
        runtime_contract=_render_runtime_contract(progress_timeout_seconds),
        spend_gate_open=spend_gate_open,
    )


def _number(value: Any) -> float | None:
    import math

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None
    return None


def _particlefield_prelaunch_spend_guard(
    *,
    allow_paid: bool,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    budget = _number(max_spend_usd)
    hourly_rate = _number(max_hourly_rate_usd)
    estimated_max_cost_usd = (
        round(float(hourly_rate) * max(0, int(max_seconds)) / 3600.0, 6)
        if hourly_rate is not None
        else None
    )
    if not allow_paid:
        blockers.append("paid_launch_not_allowed")
    if budget is None:
        blockers.append("max_spend_usd_missing")
    elif budget <= 0:
        blockers.append("max_spend_usd_not_positive")
    if hourly_rate is None:
        blockers.append("max_hourly_rate_usd_missing_or_nonfinite")
    elif hourly_rate <= 0:
        blockers.append("max_hourly_rate_usd_not_positive")
    if (
        budget is not None
        and budget > 0
        and estimated_max_cost_usd is not None
        and estimated_max_cost_usd > budget
    ):
        blockers.append("estimated_max_cost_exceeds_max_spend_usd")
    can_launch = not blockers
    return {
        "schema_version": "isaac_particlefield_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "max_spend_usd": budget,
        "budget_source": "argument" if budget is not None else "missing",
        "max_seconds": int(max_seconds),
        "max_hourly_rate_usd": hourly_rate,
        "estimated_max_cost_usd": estimated_max_cost_usd,
        "blockers": blockers,
        "claim_boundary": {
            "spend_guard_only": True,
            "can_launch_is_not_provider_success": True,
            "can_launch_is_not_render_success": True,
            "no_provider_api_call_before_can_launch": True,
        },
    }


def _teardown_proof_from_watch(provider_name: str, instance_id: str, watch: Mapping[str, Any]) -> dict:
    """Map a ``watch_and_collect`` teardown record onto a fail-closed teardown proof.

    Only an API-confirmed post-terminate state (``teardown.verification`` written by
    the watch loop from a provider state query) can prove teardown; the terminate
    call's own response is self-reported and fails the proof.
    """
    teardown = dict(watch.get("teardown") or {})
    reason = str(watch.get("teardown_reason") or "").strip()
    status = str(teardown.get("status") or "").strip().lower()
    kept_alive = status in {"stopped", "preserved", "skipped"} or reason in {
        "left_running_by_request",
        "runner_done_preserved_for_warm_reuse",
    }
    if kept_alive:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=reason or status or "kept_alive",
        )
    verification = dict(teardown.get("verification") or {})
    api_confirmed = verification.get("api_confirmed") is True
    observed_status = str(verification.get("provider_status") or "").strip().lower()
    if api_confirmed and observed_status:
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


def _write_reliability_manifest(out_dir: Path, manifest: dict) -> str:
    path = Path(out_dir) / RELIABILITY_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return str(path)


def _build_render_run_reliability_manifest(
    *,
    provider_name: str,
    run_id: str,
    session_dir: str,
    availability: Mapping[str, Any] | None,
    launch: Mapping[str, Any] | None,
    watch: Mapping[str, Any] | None,
    pre_spend_preflight: Mapping[str, Any] | None = None,
    instance_id: str = "",
) -> dict:
    """Compose the ops-facing reliability manifest for one render-lane run.

    The render lane performs no task evaluation and no artifact-quality judgment
    beyond collection, so those phases are declared not applicable rather than
    faked as PASS.
    """
    availability = dict(availability or {})
    preflight = dict(pre_spend_preflight or {})
    if not preflight:
        preflight = _phase(
            False,
            [f"capacity_unavailable:{availability.get('reason') or 'pre_spend_preflight_missing'}"],
            availability=availability,
        )

    launch_phase = None
    startup_phase = None
    runtime_phase = None
    collection_phase = None
    teardown_phase = None
    if launch is not None:
        launched = str(launch.get("status") or "") == "launched"
        launch_phase = _phase(
            launched,
            [] if launched else ["provider_launch_failed"],
            mode=launch.get("mode"),
            instance_id=launch.get("instance_id"),
        )
    if watch is not None:
        boot = dict(watch.get("last_bootstrap") or {})
        marker_seen = bool(boot)
        startup_phase = _phase(
            marker_seen,
            [] if marker_seen else ["startup_marker_timeout:no_bootstrap_marker_observed"],
            last_bootstrap_phase=str(boot.get("phase") or "") or None,
        )
        runtime_blockers: list[str] = []
        if watch.get("post_marker_progress_timeout_observed"):
            detail = dict(watch.get("post_marker_progress_timeout") or {})
            runtime_blockers.append(
                "post_marker_no_progress:"
                f"phase_{detail.get('phase') or 'unknown'}"
                f":timeout_{detail.get('timeout_seconds') or 'unknown'}s"
            )
        if watch.get("runner_timeout_observed"):
            runtime_blockers.append("runner_failed:runner_timeout")
        if watch.get("timed_out_without_runner_done"):
            runtime_blockers.append("runner_failed:collection_window_expired_without_runner_done")
        runner_result = dict(watch.get("runner_result") or {})
        runner_status = str(runner_result.get("status") or "").strip().lower()
        if runner_status == "blocked":
            runtime_blockers.append("runner_failed:runner_result_blocked")
        runtime_phase = _phase(
            marker_seen and not runtime_blockers,
            runtime_blockers if marker_seen else ["container_startup_failed_first"],
            runner_result_status=runner_status or None,
        )
        collected = bool(runner_result) and bool(watch.get("runner_result_source"))
        collection_phase = _phase(
            collected,
            [] if collected else ["artifact_collection_failed:no_runner_result_collected"],
            runner_result_source=watch.get("runner_result_source"),
        )
        teardown_phase = _teardown_proof_from_watch(provider_name, instance_id, watch)

    return build_provider_reliability_manifest(
        run_id=run_id,
        provider=provider_name,
        session_dir=session_dir,
        launched_at=utc_now_iso() if launch is not None else None,
        pre_spend_preflight=preflight,
        provider_launch=launch_phase,
        container_startup=startup_phase,
        runtime_execution=runtime_phase,
        artifact_collection=collection_phase,
        teardown=teardown_phase,
        not_applicable_phases=("artifact_quality", "task_evaluation"),
    )


def run_isaac_particlefield_render_job(
    *, source: str | Path, out_dir: str | Path, cameras: Sequence[dict] | None = None,
    cameras_file: str = "cameras.json", allow_paid: bool = False, cold: bool = False,
    image: str | None = None, key_prefix: str = "blueprint/isaac-splat", up_axis: int | None = None,
    max_seconds: int = 1200, provider: str = "runpod",
    render_options: Mapping[str, Any] | None = None,
    warm_candidates: Sequence[str] | None = None,
    preserve_instance: bool = False,
    post_marker_progress_timeout_seconds: int | None = None,
    max_spend_usd: float | None = None,
) -> dict:
    """Full job, provider-agnostic. Without ``allow_paid`` it prepares + stages and returns a
    launchable plan. ``provider`` selects the GPU backend (``runpod`` or ``vast``); the
    bundle/output transport and the watch loop are identical across providers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema_version": SCHEMA_VERSION, "status": "blocked", "blockers": [],
                      "rendered_by": "isaac_rtx_particlefield", "source": str(source),
                      "provider": (provider or "runpod").lower()}
    try:
        prov = get_render_provider(
            provider,
            warm_candidates=(
                tuple(warm_candidates) if warm_candidates is not None else DEFAULT_WARM_CANDIDATES
            ),
        )
    except ValueError as exc:
        manifest["blockers"].append("unknown_render_provider")
        manifest["error"] = str(exc)
        return manifest
    manifest["provider_available"] = prov.available()
    asset = ensure_particlefield_usd(source, out_dir / "asset")
    manifest["asset"] = asset
    if asset.get("status") != "completed":
        manifest["blockers"].extend(asset.get("blockers", ["asset_unavailable"]))
        return manifest
    if cameras is None:
        std = asset.get("standard_ply")
        if not std:
            manifest["blockers"].append("cameras_required_when_source_is_usd")
            return manifest
        cameras = derive_cameras_for(std, up_axis=up_axis)
    manifest["camera_ids"] = [c.get("id") for c in cameras]
    if render_options:
        render_options = dict(render_options)
        robot_usd = str(render_options.get("robot_usd") or "")
        robot_pose = render_options.get("robot_pose")
        if robot_usd and isinstance(robot_pose, (list, tuple)) and len(robot_pose) == 4:
            authored = author_robot_into_stage(
                asset["usdc"], robot_usd=robot_usd, robot_pose=robot_pose
            )
            manifest["robot_authoring"] = authored
            if authored.get("status") == "completed":
                render_options.setdefault("robot_prim_path", authored["robot_prim_path"])
                render_options.setdefault("lights_path", authored["lights_path"])
        manifest["render_options"] = dict(render_options)
    bundle_zip = build_render_bundle(
        usdc_path=asset["usdc"], cameras=cameras, out_dir=out_dir,
        render_options=render_options,
    )
    manifest["bundle_zip"] = str(bundle_zip)
    job_dir = out_dir / "object_store_real_run"
    staged = stage_bundle(bundle_zip, job_dir, key_prefix=key_prefix)
    manifest["staging"] = {"status": staged["status"]}
    if staged["status"] != "completed":
        manifest["blockers"].append("staging_failed")
        manifest["staging"]["stderr_tail"] = staged.get("stderr_tail")
        return manifest
    selected_image = image or default_image()
    spec = build_render_launch_spec(job_dir, image=selected_image, cameras_file=cameras_file)
    request = prov.build_request(spec, job_dir)
    manifest["launch_request_shape"] = _launch_shape_summary(prov.name, request)
    prelaunch_spend_guard = _particlefield_prelaunch_spend_guard(
        allow_paid=allow_paid,
        max_spend_usd=max_spend_usd,
        max_seconds=max_seconds,
        max_hourly_rate_usd=getattr(spec, "max_hourly_rate_usd", None),
    )
    manifest["prelaunch_spend_guard"] = prelaunch_spend_guard
    if not allow_paid:
        manifest["status"] = "prepared"
        manifest["note"] = f"staged + launchable on {prov.name}; re-run with allow_paid=True to spend GPU"
        return manifest
    if prelaunch_spend_guard.get("can_launch") is not True:
        manifest["blockers"].append("isaac_particlefield_prelaunch_spend_guard_not_passed")
        manifest["blockers"].extend(prelaunch_spend_guard.get("blockers") or [])
        manifest["provider_reliability_manifest"] = _write_reliability_manifest(
            out_dir,
            _build_render_run_reliability_manifest(
                provider_name=prov.name,
                run_id=f"prelaunch-{utc_now_iso()}",
                session_dir=str(out_dir),
                availability=None,
                launch=None,
                watch=None,
                pre_spend_preflight=_phase(
                    False,
                    [
                        "isaac_particlefield_prelaunch_spend_guard_not_passed",
                        *(prelaunch_spend_guard.get("blockers") or []),
                    ],
                    prelaunch_spend_guard=prelaunch_spend_guard,
                ),
            ),
        )
        return manifest
    progress_timeout = (
        _default_post_marker_timeout()
        if post_marker_progress_timeout_seconds is None
        else max(0, int(post_marker_progress_timeout_seconds))
    )
    avail = prov.available()
    try:
        preflight = _render_pre_spend_preflight(
            provider_name=prov.name,
            availability=avail,
            image_ref=selected_image,
            progress_timeout_seconds=progress_timeout,
            spend_gate_open=prelaunch_spend_guard.get("can_launch") is True,
        )
    except PreSpendPreflightBlocked as blocked:
        preflight = blocked.preflight
        if avail.get("reason"):
            manifest["blockers"].append(str(avail.get("reason")))
        manifest["blockers"].extend(preflight.get("blockers") or ["pre_spend_preflight_failed"])
        manifest["provider_reliability_manifest"] = _write_reliability_manifest(
            out_dir,
            _build_render_run_reliability_manifest(
                provider_name=prov.name,
                run_id=f"prelaunch-{utc_now_iso()}",
                session_dir=str(out_dir),
                availability=avail,
                launch=None,
                watch=None,
                pre_spend_preflight=preflight,
            ),
        )
        return manifest
    request["prelaunch_spend_guard"] = prelaunch_spend_guard
    # Teardown obligation is recorded BEFORE the billable call: a crash anywhere
    # between launch and collect leaves this record for reap_orphans.
    pending = open_pending_teardown(
        provider=prov.name,
        lane=RENDER_LANE,
        run_id=f"render-{int(time.time() * 1000)}",
        job_dir=str(job_dir),
        max_age_seconds=int(max_seconds) + 1800,
    )
    manifest["pending_teardown_record"] = pending["path"]
    launch = prov.launch(job_dir, request, cold=cold)
    manifest["launch"] = launch
    if launch.get("instance_id"):
        bind_pending_teardown_instance(pending["path"], str(launch["instance_id"]))
    nonce_path = job_dir / "launch_session_nonce.txt"
    run_id = ""
    if nonce_path.is_file():
        try:
            run_id = nonce_path.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            run_id = ""
    run_id = run_id or str(launch.get("instance_id") or f"launch-{utc_now_iso()}")
    if launch.get("status") != "launched":
        if not launch.get("instance_id"):
            cancel_pending_teardown(
                pending["path"], reason="launch_returned_no_allocation", evidence=launch
            )
        manifest["blockers"].append("launch_failed")
        manifest["provider_reliability_manifest"] = _write_reliability_manifest(
            out_dir,
            _build_render_run_reliability_manifest(
                provider_name=prov.name,
                run_id=run_id,
                session_dir=str(out_dir),
                availability=avail,
                launch=launch,
                watch=None,
                pre_spend_preflight=preflight,
            ),
        )
        return manifest
    result = watch_and_collect(job_dir, out_dir / "render_output", launch["instance_id"],
                               provider=prov, max_seconds=max_seconds,
                               preserve_instance=preserve_instance,
                               keep_running=preserve_instance,
                               progress_timeout_seconds=progress_timeout)
    manifest["render"] = result
    reliability = _build_render_run_reliability_manifest(
        provider_name=prov.name,
        run_id=run_id,
        session_dir=str(out_dir),
        availability=avail,
        launch=launch,
        watch=result,
        pre_spend_preflight=preflight,
        instance_id=str(launch.get("instance_id") or ""),
    )
    manifest["provider_reliability_manifest"] = _write_reliability_manifest(
        out_dir, reliability
    )
    # Only a PASSING teardown proof can close the pending record; anything else
    # stays open for the orphan reaper.
    close_pending_teardown(
        pending["path"],
        dict((reliability.get("phase_contracts") or {}).get("teardown") or {}),
    )
    manifest["status"] = "completed" if result.get("status") == "completed" else "blocked"
    return manifest


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Isaac ParticleField splat render job")
    ap.add_argument("--source", help="capture splat (.ply/.spz) or ParticleField .usdc")
    ap.add_argument("--out-dir")
    ap.add_argument("--cameras-file", default="cameras.json", choices=["cameras.json", "cameras_canary.json"])
    ap.add_argument("--provider", default="runpod", choices=["runpod", "vast", "digitalocean", "gcp", "aws"],
                    help="GPU backend (provider-agnostic launch; bundle/watch identical)")
    ap.add_argument("--allow-paid", action="store_true", help="spend GPU (otherwise prepare+stage only)")
    ap.add_argument("--cold", action="store_true", help="force cold on-demand create")
    ap.add_argument("--image", default=None)
    ap.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2])
    ap.add_argument("--max-seconds", type=int, default=1200)
    ap.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="Required with --allow-paid; blocks before provider launch when absent.",
    )
    ap.add_argument("--list-providers", action="store_true",
                    help="print providers + credential availability and exit")
    args = ap.parse_args(argv)
    if args.list_providers:
        print(json.dumps(list_render_providers(), indent=2))
        return 0
    if not args.source or not args.out_dir:
        ap.error("--source and --out-dir are required (unless --list-providers)")
    m = run_isaac_particlefield_render_job(
        source=args.source, out_dir=args.out_dir, cameras_file=args.cameras_file,
        allow_paid=args.allow_paid, cold=args.cold, image=args.image, up_axis=args.up_axis,
        max_seconds=args.max_seconds, provider=args.provider, max_spend_usd=args.max_spend_usd,
    )
    print(json.dumps(m, indent=2, default=str))
    return 0 if m.get("status") in ("completed", "prepared") else 1


if __name__ == "__main__":
    raise SystemExit(main())
