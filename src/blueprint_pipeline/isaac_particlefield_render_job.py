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
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gpu_render_providers import RenderLaunchSpec, get_render_provider, list_render_providers

SCHEMA_VERSION = "isaac_particlefield_render_job.v1"
SECRETS = Path.home() / ".blueprint-secrets"
DEFAULT_WARM_CANDIDATES = (
    "pwbu7wxsvxpr0x", "9zxerj0nm3ow76", "qzgtsh4t27hi7f", "v4bd9u2qhwivb8",
    "y3n5n7t6wvaawe", "1gx9uri0mkrxg9", "ajxbj2ysyow3n9", "usjvua1bwlwhyj",
)
_SPLAT_SUFFIXES = {".ply", ".spz", ".usdc", ".usd", ".usda", ".usdz"}

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
mark("runner_done", rc=rc)
# Keep the container process alive after runner completion so RunPod does not restart it and
# clobber the final output object before the parent collector observes runner_done.
while True:
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

def _env_for(bundle_url: str, put_url: str, cameras_file: str, *, width: int, height: int,
             subframes: int, rt_subframes: int, warmup: int) -> dict:
    return {
        "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "CUDA_VISIBLE_DEVICES": "0",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "CAMERAS_FILE": cameras_file,
        "RENDER_WIDTH": str(width), "RENDER_HEIGHT": str(height),
        "RENDER_SUBFRAMES": str(subframes), "RENDER_RT_SUBFRAMES": str(rt_subframes),
        "RENDER_WARMUP_FRAMES": str(warmup),
    }


def build_render_launch_spec(
    job_dir: Path, *, image: str, cameras_file: str, width: int = 1280, height: int = 960,
    subframes: int = 8, rt_subframes: int = 64, warmup: int = 30, container_disk_gb: int = 140,
    volume_gb: int = 80, gpu_types: Sequence[str] = ("NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"),
    max_hourly_rate_usd: float = 2.0, min_gpu_ram_mb: int = 24000,
) -> RenderLaunchSpec:
    """Provider-neutral render launch spec (image + env + bootstrap + GPU sizing). The
    env carries the signed bundle GET + output PUT URLs, so any provider just forwards it."""
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text().strip()
    put_url = (job_dir / "provider_output_put_url.txt").read_text().strip()
    env = _env_for(bundle_url, put_url, cameras_file, width=width, height=height,
                   subframes=subframes, rt_subframes=rt_subframes, warmup=warmup)
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
                  warm_candidates: Sequence[str] = DEFAULT_WARM_CANDIDATES) -> dict:
    """Backward-compat RunPod launch (warm-restart-then-cold). Delegates to the provider
    abstraction. Returns {instance_id, pod_id, mode, attempts}; spends money."""
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


def _runner_result_from_dir(out_dir: Path) -> tuple[dict, str | None]:
    for name in ("isaac_runtime_result.json", "isaac_g1_kitchen_parity_result.json"):
        payload = _read_json_file(out_dir / name)
        if payload:
            return payload, name
    return {}, None


def watch_and_collect(job_dir: Path, out_dir: Path, instance_id: str, *, provider=None,
                      max_seconds: int = 1200, poll: int = 25,
                      stop_on_success: bool = True,
                      preserve_instance: bool = False,
                      keep_running: bool = False) -> dict:
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
    expected_launch_session_id = ""
    nonce_path = job_dir / "launch_session_nonce.txt"
    if nonce_path.is_file():
        try:
            expected_launch_session_id = nonce_path.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            expected_launch_session_id = ""
    while time.time() - t0 < max_seconds:
        time.sleep(poll)
        try:
            data = urllib.request.urlopen(get_url, timeout=60).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(out_dir)
        except Exception:  # noqa: BLE001
            continue
        last, last_source = _runner_result_from_dir(out_dir)
        boot = _read_json_file(out_dir / "bootstrap.json")
        if boot:
            last_boot = boot
        console = out_dir / "runner_console.log"
        if console.is_file():
            try:
                last_console_tail = console.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:  # noqa: BLE001
                pass
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
        try:
            data = urllib.request.urlopen(get_url, timeout=60).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(out_dir)
        except Exception:  # noqa: BLE001
            pass
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
    should_preserve_for_warm_reuse = bool(
        done
        and runner_done_observed
        and not done_from_final_result_without_runner_done
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
        teardown_reason = (
            "runner_timeout_terminated"
            if runner_timeout_observed
            else
            "final_result_without_runner_done_terminated"
            if done_from_final_result_without_runner_done
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
            "runner_done_observed": runner_done_observed,
            "runner_timeout_observed": runner_timeout_observed,
            "final_result_without_runner_done": done_from_final_result_without_runner_done,
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


def run_isaac_particlefield_render_job(
    *, source: str | Path, out_dir: str | Path, cameras: Sequence[dict] | None = None,
    cameras_file: str = "cameras.json", allow_paid: bool = False, cold: bool = False,
    image: str | None = None, key_prefix: str = "blueprint/isaac-splat", up_axis: int | None = None,
    max_seconds: int = 1200, provider: str = "runpod",
    render_options: Mapping[str, Any] | None = None,
    warm_candidates: Sequence[str] | None = None,
    preserve_instance: bool = False,
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
    spec = build_render_launch_spec(job_dir, image=image or default_image(), cameras_file=cameras_file)
    request = prov.build_request(spec, job_dir)
    manifest["launch_request_shape"] = _launch_shape_summary(prov.name, request)
    if not allow_paid:
        manifest["status"] = "prepared"
        manifest["note"] = f"staged + launchable on {prov.name}; re-run with allow_paid=True to spend GPU"
        return manifest
    avail = prov.available()
    if not avail.get("available"):
        manifest["blockers"].append(avail.get("reason") or "provider_credentials_missing")
        return manifest
    launch = prov.launch(job_dir, request, cold=cold)
    manifest["launch"] = launch
    if launch.get("status") != "launched":
        manifest["blockers"].append("launch_failed")
        return manifest
    result = watch_and_collect(job_dir, out_dir / "render_output", launch["instance_id"],
                               provider=prov, max_seconds=max_seconds,
                               preserve_instance=preserve_instance,
                               keep_running=preserve_instance)
    manifest["render"] = result
    manifest["status"] = "completed" if result.get("status") == "completed" else "blocked"
    return manifest


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Isaac ParticleField splat render job")
    ap.add_argument("--source", help="capture splat (.ply/.spz) or ParticleField .usdc")
    ap.add_argument("--out-dir")
    ap.add_argument("--cameras-file", default="cameras.json", choices=["cameras.json", "cameras_canary.json"])
    ap.add_argument("--provider", default="runpod", choices=["runpod", "vast"],
                    help="GPU backend (provider-agnostic launch; bundle/watch identical)")
    ap.add_argument("--allow-paid", action="store_true", help="spend GPU (otherwise prepare+stage only)")
    ap.add_argument("--cold", action="store_true", help="force cold on-demand create")
    ap.add_argument("--image", default=None)
    ap.add_argument("--up-axis", type=int, default=None, choices=[0, 1, 2])
    ap.add_argument("--max-seconds", type=int, default=1200)
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
        max_seconds=args.max_seconds, provider=args.provider,
    )
    print(json.dumps(m, indent=2, default=str))
    return 0 if m.get("status") in ("completed", "prepared") else 1


if __name__ == "__main__":
    raise SystemExit(main())
