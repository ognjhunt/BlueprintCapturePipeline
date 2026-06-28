"""End-to-end MuJoCo-parity G1 kitchen eval on Isaac Sim + GPU (productionized).

Replicates the MuJoCo G1 walk-to-target eval (policy + per-step trace + outcome + WAM-ready
package + MP4) but executes it inside Isaac Sim on a real GPU, against the sim-ready Lightwheel
kitchen USD and the official Isaac G1 USD. Stage A drives the SAME deterministic controller as
MuJoCo (``isaac_g1_policy``); Stage B swaps in GR00T N1.7 SONIC via ``--policy groot_sonic``.

Chain:
  scenarios + kitchen asset dir + G1 USD ref
    -> bundle (runner + policy module + request.json + kitchen assets)
    -> stage to object store (signed GET/PUT)        [reused from the splat job]
    -> provider GPU pod runs run_isaac_g1_kitchen_parity_eval.py via a hardened bootstrap
       (provider-agnostic: RunPod or Vast)            [reused gpu_render_providers]
    -> RTX MP4s + traces + parity outcome JSON uploaded
    -> collect -> assemble the WAM-ready harness package with an honest claim boundary.

Paid GPU launches are gated behind ``allow_paid=True``. Secrets are file-based and never
logged. Truth boundary: Isaac RTX kinematic walk-to-target preview parity (Stage A) — not
dynamic locomotion, not a learned policy, not readiness.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .isaac_particlefield_render_job import (
    DEFAULT_WARM_CANDIDATES, _read_secret, default_image, stage_bundle, watch_and_collect,
)

SCHEMA_VERSION = "isaac_g1_kitchen_parity_job.v1"
WORKER_BUNDLE_DIR = "/workspace/bundle"
DEFAULT_G1_USD_RELATIVE = "Isaac/Robots/Unitree/G1/g1.usd"
DEFAULT_KITCHEN_MAIN_USD = "Collected_KitchenRoom/KitchenRoom.usd"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _zip_dir(src_dir: Path, zip_path: Path) -> Path:
    """Zip a directory tree (files only), paths relative to src_dir."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(src_dir.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(src_dir).as_posix())
    return zip_path


def parity_image() -> str:
    """Isaac worker image for the parity eval (defaults to the same Isaac eval worker)."""
    ref = _read_secret("isaac_eval_worker_image_ref")
    return ref or default_image()


# diagnostics-streaming pod bootstrap for the parity runner
BOOTSTRAP = r'''
import os, sys, io, time, json, zipfile, threading, subprocess, urllib.request, pathlib
OUT="/workspace/out"; BUNDLE="/workspace/bundle"
for d in (OUT, BUNDLE): pathlib.Path(d).mkdir(parents=True, exist_ok=True)
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
GETB=os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI","")
def putout():
    try:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in pathlib.Path(OUT).rglob("*"):
                if p.is_file():
                    try: z.write(p, p.relative_to(OUT).as_posix())
                    except Exception: pass
        req=urllib.request.Request(PUT, data=buf.getvalue(), method="PUT", headers={"Content-Type":"application/zip"})
        urllib.request.urlopen(req, timeout=180).read()
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
# kitchen assets are staged separately (large, reused across iterations) and fetched into BUNDLE/kitchen
KURL=os.environ.get("KITCHEN_BUNDLE_URL","")
if KURL:
    mark("kitchen_fetching")
    kdir=BUNDLE+"/kitchen"; pathlib.Path(kdir).mkdir(parents=True, exist_ok=True)
    kdata=urllib.request.urlopen(KURL, timeout=1800).read()
    zipfile.ZipFile(io.BytesIO(kdata)).extractall(kdir)
    mark("kitchen_extracted", kitchen_files=len(list(pathlib.Path(kdir).rglob("*"))))
threading.Thread(target=hb, daemon=True).start()
try: subprocess.call(["/isaac-sim/python.sh","-m","pip","install","-q","pillow"])  # frame save dep (best-effort)
except Exception: pass
try: subprocess.call(["bash","-c","command -v ffmpeg >/dev/null 2>&1 || (apt-get update -y >/dev/null 2>&1 && apt-get install -y ffmpeg >/dev/null 2>&1)"])  # mp4 assembly (best-effort)
except Exception: pass
cmd=["/isaac-sim/python.sh", BUNDLE+"/run_isaac_g1_kitchen_parity_eval.py",
     "--request", BUNDLE+"/request.json", "--out-dir", OUT,
     "--policy", os.environ.get("PARITY_POLICY","blueprint_default_walk_to_target_smoke_policy"),
     "--steps", os.environ.get("PARITY_STEPS","64"),
     "--width", os.environ.get("RENDER_WIDTH","1280"), "--height", os.environ.get("RENDER_HEIGHT","960"),
     "--fps", os.environ.get("RENDER_FPS","20"),
     "--warmup-frames", os.environ.get("RENDER_WARMUP","6"),
     "--per-scenario-seconds", os.environ.get("PARITY_PER_SCENARIO_SECONDS","420"),
     "--focus-radius", os.environ.get("PARITY_FOCUS_RADIUS","0"),
     "--settle-seconds", os.environ.get("PARITY_SETTLE_SECONDS","0")]
if os.environ.get("PARITY_KEEP_OBJECTS",""): cmd += ["--keep-objects", os.environ["PARITY_KEEP_OBJECTS"]]
if os.environ.get("PARITY_NO_PROBE","")=="1": cmd.append("--no-collision-probe")
if os.environ.get("PARITY_CHEAP_COLLISION","")=="1": cmd.append("--cheap-collision")
if os.environ.get("PARITY_ARTICULATED","")=="1": cmd.append("--articulated")
if os.environ.get("PARITY_PHYSICS_ARTICULATION_DRIVE","")=="1": cmd.append("--physics-articulation-drive")
if os.environ.get("PARITY_DYNAMIC_STANDING_CONTACT_STEPS",""): cmd += ["--dynamic-standing-contact-steps", os.environ["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"]]
if os.environ.get("PARITY_MANIPULATION_CAM","")=="1": cmd.append("--manipulation-cam")
if os.environ.get("PARITY_MANIPULATION_LOOK_AT",""): cmd += ["--manipulation-look-at", os.environ["PARITY_MANIPULATION_LOOK_AT"]]
if os.environ.get("PARITY_RENDER_SUBFRAMES",""): cmd += ["--render-subframes", os.environ["PARITY_RENDER_SUBFRAMES"]]
if os.environ.get("PARITY_MANIPULATION_REACH","")=="1": cmd.append("--manipulation-reach")
if os.environ.get("PARITY_MANIPULATION_REACH_ARM",""): cmd += ["--manipulation-reach-arm", os.environ["PARITY_MANIPULATION_REACH_ARM"]]
if os.environ.get("PARITY_FILL_LIGHT_INTENSITY",""): cmd += ["--fill-light-intensity", os.environ["PARITY_FILL_LIGHT_INTENSITY"]]
if os.environ.get("PARITY_NEUTRAL_ENVIRONMENT","")=="1": cmd.append("--neutral-environment")
if os.environ.get("PARITY_COLLISION_APPROXIMATION",""): cmd += ["--collision-approximation", os.environ["PARITY_COLLISION_APPROXIMATION"]]
if os.environ.get("PARITY_VERIFY_CAM","")=="1": cmd.append("--verify-cam")
if os.environ.get("PARITY_MANIPULATION_STAND","")=="1": cmd.append("--manipulation-stand")
if os.environ.get("PARITY_KINEMATIC_ARM_POSE","")=="1": cmd.append("--kinematic-arm-pose")
mark("runner_starting", cmd=cmd)
rc=subprocess.call(cmd)
mark("runner_done", rc=rc)
# Keep the container process alive after runner completion so RunPod does not restart it and
# clobber the final output object before the parent collector observes runner_done.
while True:
    time.sleep(30); putout()
'''

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
    script = (
        "set +e\n"
        "mkdir -p /workspace/out\n"
        "cat > /workspace/early.py <<'EARLYEOF'\n" + _EARLY_MARKER + "\nEARLYEOF\n"
        "(python3 /workspace/early.py 2>/dev/null || /isaac-sim/python.sh /workspace/early.py 2>/dev/null) || true\n"
        "cat > /workspace/boot.py <<'PYEOF'\n" + BOOTSTRAP + "\nPYEOF\n"
        "/isaac-sim/python.sh /workspace/boot.py 2>&1 | tee /workspace/out/runner_console.log\n"
    )
    return ["-lc", script]


# ----------------------------- request + bundle -----------------------------

def build_request(*, scenarios: Sequence[dict], kitchen_main_usd_relative: str = DEFAULT_KITCHEN_MAIN_USD,
                  g1_usd: str = DEFAULT_G1_USD_RELATIVE, policy_id: str, steps: int) -> dict:
    """The runner's request.json. kitchen_usd is the worker-absolute path inside the extracted
    bundle; g1_usd is a relative Isaac asset path resolved against the assets root on the worker."""
    return {
        "schema_version": "isaac_g1_kitchen_parity_request.v1",
        "kitchen_usd": f"{WORKER_BUNDLE_DIR}/kitchen/{kitchen_main_usd_relative}",
        "g1_usd": g1_usd,
        "policy_id": policy_id,
        "steps": steps,
        "scenarios": list(scenarios),
    }


def build_parity_bundle(*, scenarios: Sequence[dict], out_dir: Path,
                        kitchen_asset_dir: str | Path | None = None,
                        kitchen_main_usd_relative: str = DEFAULT_KITCHEN_MAIN_USD,
                        g1_usd: str = DEFAULT_G1_USD_RELATIVE,
                        policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
                        steps: int = 64) -> Path:
    """Assemble the GPU bundle: the runner + the policy module + request.json + (optional) the
    kitchen asset tree under kitchen/. The runner imports the shipped policy module on the worker."""
    bundle = out_dir / "bundle"
    (bundle / "kitchen").mkdir(parents=True, exist_ok=True)
    runner = _repo_root() / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"
    policy = _repo_root() / "src" / "blueprint_pipeline" / "isaac_g1_policy.py"
    (bundle / "run_isaac_g1_kitchen_parity_eval.py").write_bytes(runner.read_bytes())
    (bundle / "isaac_g1_policy.py").write_bytes(policy.read_bytes())
    # Ship the scene_placement package alongside the runner so its dynamic task->object resolution
    # works on the worker (the runner imports `scene_placement` from the bundle dir; it falls back to
    # the repo's `blueprint_pipeline.scene_placement` in tests). Without this the worker has no
    # `blueprint_pipeline` on its path and the runner silently degrades to the scenario's literal
    # target. The package is pure-Python + intra-package imports only, so a flat copy is importable.
    import shutil
    sp_src = _repo_root() / "src" / "blueprint_pipeline" / "scene_placement"
    sp_dst = bundle / "scene_placement"
    if sp_dst.exists():
        shutil.rmtree(sp_dst)
    shutil.copytree(sp_src, sp_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    request = build_request(scenarios=scenarios, kitchen_main_usd_relative=kitchen_main_usd_relative,
                            g1_usd=g1_usd, policy_id=policy_id, steps=steps)
    (bundle / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    if kitchen_asset_dir is not None:
        src = Path(kitchen_asset_dir)
        for item in src.rglob("*"):
            if item.is_file():
                dst = bundle / "kitchen" / item.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(item.read_bytes())
    zip_path = out_dir / "isaac_g1_kitchen_parity_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(bundle.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(bundle).as_posix())
    return zip_path


def build_launch_spec(job_dir: Path, *, image: str, policy_id: str, steps: int, width: int = 1280,
                      height: int = 960, fps: int = 20, container_disk_gb: int = 140,
                      volume_gb: int = 80, kitchen_url: str | None = None, warmup: int = 6,
                      per_scenario_seconds: int = 420, no_collision_probe: bool = False,
                      focus_radius: float = 0.0, keep_objects: str = "", settle_seconds: int = 0,
                      cheap_collision: bool = False, articulated: bool = False,
                      physics_articulation_drive: bool = False,
                      dynamic_standing_contact_steps: int = 0,
                      manipulation_cam: bool = False, manipulation_look_at: str = "",
                      render_subframes: int = 0, manipulation_reach: bool = False,
                      manipulation_reach_arm: str = "both",
                      fill_light_intensity: float = 0.0,
                      neutral_environment: bool = False,
                      kinematic_arm_pose: bool = False,
                      collision_approximation: str = "",
                      verify_cam: bool = False,
                      manipulation_stand: bool = False) -> RenderLaunchSpec:
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text().strip()
    put_url = (job_dir / "provider_output_put_url.txt").read_text().strip()
    env = {
        "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "PARITY_POLICY": policy_id, "PARITY_STEPS": str(steps),
        "RENDER_WIDTH": str(width), "RENDER_HEIGHT": str(height), "RENDER_FPS": str(fps),
        "RENDER_WARMUP": str(warmup), "PARITY_PER_SCENARIO_SECONDS": str(per_scenario_seconds),
    }
    if no_collision_probe:
        env["PARITY_NO_PROBE"] = "1"
    if focus_radius and focus_radius > 0:
        env["PARITY_FOCUS_RADIUS"] = str(focus_radius)
    if keep_objects:
        env["PARITY_KEEP_OBJECTS"] = keep_objects
    if settle_seconds and settle_seconds > 0:
        env["PARITY_SETTLE_SECONDS"] = str(settle_seconds)
    if cheap_collision:
        env["PARITY_CHEAP_COLLISION"] = "1"
    if articulated or physics_articulation_drive or dynamic_standing_contact_steps > 0:
        env["PARITY_ARTICULATED"] = "1"
    if physics_articulation_drive or dynamic_standing_contact_steps > 0:
        env["PARITY_PHYSICS_ARTICULATION_DRIVE"] = "1"
    if dynamic_standing_contact_steps > 0:
        env["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"] = str(int(dynamic_standing_contact_steps))
    if manipulation_cam:
        env["PARITY_MANIPULATION_CAM"] = "1"
    if manipulation_look_at:
        env["PARITY_MANIPULATION_LOOK_AT"] = str(manipulation_look_at)
    if render_subframes and render_subframes > 0:
        env["PARITY_RENDER_SUBFRAMES"] = str(render_subframes)
    if manipulation_reach:
        env["PARITY_MANIPULATION_REACH"] = "1"
    if manipulation_reach and manipulation_reach_arm:
        env["PARITY_MANIPULATION_REACH_ARM"] = str(manipulation_reach_arm)
    if fill_light_intensity and fill_light_intensity > 0:
        env["PARITY_FILL_LIGHT_INTENSITY"] = str(fill_light_intensity)
    if neutral_environment:
        env["PARITY_NEUTRAL_ENVIRONMENT"] = "1"
    if kinematic_arm_pose:
        env["PARITY_KINEMATIC_ARM_POSE"] = "1"
    if collision_approximation:
        env["PARITY_COLLISION_APPROXIMATION"] = str(collision_approximation)
    if verify_cam:
        env["PARITY_VERIFY_CAM"] = "1"
    if manipulation_stand:
        env["PARITY_MANIPULATION_STAND"] = "1"
    if kitchen_url:
        env["KITCHEN_BUNDLE_URL"] = kitchen_url
    return RenderLaunchSpec(
        name="blueprint-isaac-g1-kitchen-parity", image=image, env=env,
        bootstrap_argv=docker_start_cmd(), entrypoint=["bash"],
        container_disk_gb=container_disk_gb, volume_gb=volume_gb,
    )


# ----------------------------- WAM-ready harness package -----------------------------

def build_harness_package(*, result: dict, render_out_dir: Path, out_dir: Path) -> dict:
    """Assemble the WAM-ready harness package from the collected Isaac run: per-scenario MP4s +
    traces + outcome become the inputs the WAM video-fidelity evaluator consumes. Running the
    OSCAR/COSMOS model itself is a separate GPU/checkpoint-gated step; this packages its inputs
    and records the honest claim boundary."""
    scenarios = result.get("scenarios", []) if isinstance(result, dict) else []
    items = []
    for sc in scenarios:
        sid = sc.get("scenario_id")
        sdir = render_out_dir / str(sid)
        items.append({
            "scenario_id": sid,
            "task_success": sc.get("task_success"),
            "trace_jsonl": str(sdir / "trace.jsonl"),
            "overview_mp4": str(sdir / "overview.mp4"),
            "robot_pov_mp4": str(sdir / "robot_pov.mp4"),
            "wam_reference_video": str(sdir / "overview.mp4"),
        })
    package = {
        "schema_version": "isaac_g1_kitchen_parity_harness.v1",
        "policy_id": result.get("policy_id") if isinstance(result, dict) else None,
        "rendered_by_isaac_rtx": True,
        "scenarios_executed": result.get("scenarios_executed") if isinstance(result, dict) else 0,
        "scenarios_passed": result.get("scenarios_passed") if isinstance(result, dict) else 0,
        "wam_evaluator": {
            "evaluator_id": "oscar_cosmos_wam_evaluator",
            "evaluates": "video_rollout_fidelity_not_task_success",
            "status": "inputs_ready_pending_model_run",
            "inputs": items,
        },
        "claim_boundary": (
            "Isaac RTX kinematic walk-to-target preview parity with the MuJoCo lane. The WAM "
            "evaluator judges generated-rollout video fidelity, not task success or readiness. "
            "Task success is the deterministic outcome contract in the result."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "isaac_g1_kitchen_parity_harness.json").write_text(json.dumps(package, indent=2))
    return package


# ----------------------------- launch with flaky-pod retry -----------------------------

def launch_with_marker_retry(prov, job_dir: Path, request: dict, *, max_attempts: int = 3,
                             marker_timeout: int = 150, poll: int = 15) -> dict:
    """Cold-create a pod, then wait for its container's early heartbeat (``bootstrap.json`` on the
    output URL). RunPod cold pods are ~50% flaky — created + billing but the container never runs.
    If no marker appears within ``marker_timeout``, terminate that pod and retry, so we never pay
    for a dead pod. Returns the launch of the first pod that actually started."""
    get_url = (job_dir / "provider_output_get_url.txt").read_text().strip()
    attempts: list[dict] = []
    for attempt in range(max_attempts):
        launch = prov.launch(job_dir, request, cold=True)
        if launch.get("status") != "launched":
            attempts.append({"attempt": attempt, "result": "launch_call_failed", "detail": launch})
            continue
        iid = launch["instance_id"]
        t0 = time.time()
        marker_seen = False
        while time.time() - t0 < marker_timeout:
            time.sleep(poll)
            try:
                data = urllib.request.urlopen(get_url, timeout=60).read()
                if "bootstrap.json" in zipfile.ZipFile(io.BytesIO(data)).namelist():
                    marker_seen = True
                    break
            except Exception:  # noqa: BLE001
                continue
        attempts.append({"attempt": attempt, "instance_id": iid, "marker_seen": marker_seen})
        if marker_seen:
            return {"status": "launched", "instance_id": iid, "mode": "cold_create_marker_verified",
                    "attempts": attempts}
        prov.terminate(iid)  # flaky pod (billing but not running) -> kill and retry
    return {"status": "blocked", "blockers": ["all_cold_launch_attempts_flaky"], "attempts": attempts}


# ----------------------------- orchestration -----------------------------

def run_isaac_g1_kitchen_parity_job(
    *, scenarios: Sequence[dict], out_dir: str | Path, kitchen_asset_dir: str | Path | None = None,
    kitchen_url: str | None = None,
    g1_usd: str = DEFAULT_G1_USD_RELATIVE, policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
    steps: int = 64, provider: str = "runpod", allow_paid: bool = False, cold: bool = False,
    image: str | None = None, key_prefix: str = "blueprint/isaac-g1-parity", max_seconds: int = 1500,
    width: int = 1280, height: int = 960, warmup: int = 6, per_scenario_seconds: int = 420,
    no_collision_probe: bool = False, focus_radius: float = 0.0, keep_objects: str = "",
    settle_seconds: int = 0, cheap_collision: bool = False, articulated: bool = False,
    physics_articulation_drive: bool = False,
    dynamic_standing_contact_steps: int = 0,
    manipulation_cam: bool = False, manipulation_look_at: str = "", render_subframes: int = 0,
    manipulation_reach: bool = False, manipulation_reach_arm: str = "both",
    fill_light_intensity: float = 0.0,
    neutral_environment: bool = False,
    kinematic_arm_pose: bool = False,
    collision_approximation: str = "",
    verify_cam: bool = False,
    manipulation_stand: bool = False,
) -> dict:
    """Full parity job. Without ``allow_paid`` it bundles + stages and returns a launchable plan."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema_version": SCHEMA_VERSION, "status": "blocked", "blockers": [],
                      "provider": (provider or "runpod").lower(), "policy_id": policy_id,
                      "rendered_by": "isaac_rtx_g1_kitchen_parity"}
    try:
        prov = get_render_provider(provider, warm_candidates=DEFAULT_WARM_CANDIDATES)
    except ValueError as exc:
        manifest["blockers"].append("unknown_render_provider")
        manifest["error"] = str(exc)
        return manifest
    manifest["provider_available"] = prov.available()
    if not scenarios:
        manifest["blockers"].append("no_scenarios")
        return manifest
    manifest["scenario_ids"] = [s.get("scenario_id") or s.get("id") for s in scenarios]
    # Stage the large kitchen tree ONCE (reused across iterations); keep the code bundle tiny.
    # A caller may pass a previously-staged kitchen_url to skip the 1.2GB re-upload entirely.
    if kitchen_url:
        manifest["kitchen_staging"] = {"status": "reused_existing_url"}
    elif kitchen_asset_dir is not None:
        kzip = out_dir / "kitchen_assets.zip"
        _zip_dir(Path(kitchen_asset_dir), kzip)
        kjob = out_dir / "kitchen_object_store"
        kstaged = stage_bundle(kzip, kjob, key_prefix=key_prefix + "/kitchen")
        manifest["kitchen_staging"] = {"status": kstaged["status"], "zip_bytes": kzip.stat().st_size}
        if kstaged["status"] != "completed":
            manifest["blockers"].append("kitchen_staging_failed")
            manifest["kitchen_staging"]["stderr_tail"] = kstaged.get("stderr_tail")
            return manifest
        kitchen_url = (kjob / "provider_bundle_url.txt").read_text().strip()
    manifest["kitchen_assets_shipped"] = kitchen_url is not None
    bundle_zip = build_parity_bundle(scenarios=scenarios, out_dir=out_dir,
                                     kitchen_asset_dir=None, g1_usd=g1_usd,
                                     policy_id=policy_id, steps=steps)
    manifest["bundle_zip"] = str(bundle_zip)
    job_dir = out_dir / "object_store_real_run"
    staged = stage_bundle(bundle_zip, job_dir, key_prefix=key_prefix)
    manifest["staging"] = {"status": staged["status"]}
    if staged["status"] != "completed":
        manifest["blockers"].append("staging_failed")
        manifest["staging"]["stderr_tail"] = staged.get("stderr_tail")
        return manifest
    spec = build_launch_spec(job_dir, image=image or parity_image(), policy_id=policy_id,
                             steps=steps, kitchen_url=kitchen_url, width=width, height=height,
                             warmup=warmup, per_scenario_seconds=per_scenario_seconds,
                             no_collision_probe=no_collision_probe, focus_radius=focus_radius,
                             keep_objects=keep_objects, settle_seconds=settle_seconds,
                             cheap_collision=cheap_collision, articulated=articulated,
                             physics_articulation_drive=physics_articulation_drive,
                             dynamic_standing_contact_steps=dynamic_standing_contact_steps,
                             manipulation_cam=manipulation_cam, manipulation_look_at=manipulation_look_at,
                             render_subframes=render_subframes, manipulation_reach=manipulation_reach,
                             manipulation_reach_arm=manipulation_reach_arm,
                             fill_light_intensity=fill_light_intensity,
                             neutral_environment=neutral_environment,
                             kinematic_arm_pose=kinematic_arm_pose,
                             collision_approximation=collision_approximation, verify_cam=verify_cam,
                             manipulation_stand=manipulation_stand)
    request_body = prov.build_request(spec, job_dir)
    manifest["launch_request_shape"] = {"provider": prov.name, "image": spec.image,
                                        "policy_id": policy_id, "steps": steps,
                                        "physics_articulation_drive": bool(
                                            physics_articulation_drive
                                            or dynamic_standing_contact_steps > 0
                                        ),
                                        "dynamic_standing_contact_steps": int(
                                            dynamic_standing_contact_steps
                                        )}
    if not allow_paid:
        manifest["status"] = "prepared"
        manifest["note"] = f"bundled + staged + launchable on {prov.name}; re-run with allow_paid=True to spend GPU"
        return manifest
    avail = prov.available()
    if not avail.get("available"):
        manifest["blockers"].append(avail.get("reason") or "provider_credentials_missing")
        return manifest
    # cold ~10-15GB Isaac image pulls on congested nodes routinely exceed 150s before the container
    # starts bash; give the early marker a generous window (+ an extra attempt) so a slow pull is not
    # mistaken for a dead pod (which caused all-dud batches on both providers).
    launch = launch_with_marker_retry(prov, job_dir, request_body, marker_timeout=420, max_attempts=4)
    manifest["launch"] = launch
    if launch.get("status") != "launched":
        manifest["blockers"].append("launch_failed_all_attempts_flaky")
        return manifest
    render_out = out_dir / "render_output"
    result = watch_and_collect(job_dir, render_out, launch["instance_id"], provider=prov,
                               max_seconds=max_seconds)
    manifest["render"] = {
        "status": result.get("status"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "teardown": result.get("teardown"),
        "runner_result_source": result.get("runner_result_source"),
        "last_bootstrap": result.get("last_bootstrap"),
        "runner_console_tail": result.get("runner_console_tail"),
    }
    parity_result = result.get("runner_result") or {}
    try:
        parity_result = json.loads((render_out / "isaac_g1_kitchen_parity_result.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    manifest["parity_result"] = parity_result
    if parity_result.get("status") == "completed":
        manifest["harness"] = build_harness_package(result=parity_result, render_out_dir=render_out,
                                                    out_dir=out_dir)
        manifest["status"] = "completed"
    else:
        manifest["blockers"].append("isaac_runtime_did_not_complete")
    return manifest


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Isaac G1 kitchen MuJoCo-parity eval job (GPU)")
    ap.add_argument("--scenarios", required=True, help="JSON file: list of {scenario_id, spawn_position_xyz, target_position_xyz}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kitchen-asset-dir", default=None, help="local Collected_KitchenRoom parent dir to ship in the bundle")
    ap.add_argument("--g1-usd", default=DEFAULT_G1_USD_RELATIVE)
    ap.add_argument("--policy", default="blueprint_default_walk_to_target_smoke_policy",
                    choices=["blueprint_default_walk_to_target_smoke_policy", "groot_sonic"])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--provider", default="runpod", choices=["runpod", "vast"])
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-seconds", type=int, default=1500)
    ap.add_argument("--articulated", action="store_true")
    ap.add_argument("--physics-articulation-drive", action="store_true")
    ap.add_argument("--dynamic-standing-contact-steps", type=int, default=0)
    ap.add_argument("--cheap-collision", action="store_true")
    ap.add_argument("--settle-seconds", type=int, default=0)
    ap.add_argument("--manipulation-cam", action="store_true",
                    help="egocentric at-sink POV (manipulation framing) instead of the navigation chase cam")
    ap.add_argument("--manipulation-look-at", default="",
                    help="fixed world 'x,y,z' the manipulation cam aims at (e.g. the faucet)")
    ap.add_argument("--render-subframes", type=int, default=0,
                    help="RTX subframes accumulated per captured frame to denoise grain (e.g. 16)")
    ap.add_argument("--manipulation-reach", action="store_true",
                    help="animate the arm reaching the faucet so the skeleton-video encodes the task")
    ap.add_argument("--manipulation-reach-arm", default="both", choices=["right", "left", "both"])
    ap.add_argument("--fill-light-intensity", type=float, default=0.0,
                    help="sphere fill light over the faucet workspace to lift the dark basin (0=off)")
    ap.add_argument("--neutral-environment", action="store_true",
                    help="replace the kitchen's outdoor-HDRI dome with a neutral environment "
                         "(no cityscape through the windows + lifts shadows)")
    ap.add_argument("--kinematic-arm-pose", action="store_true",
                    help="pose the rendered arm reaching the workspace (pure-USD, crash-safe)")
    ap.add_argument("--collision-approximation", default="",
                    choices=["", "boundingCube", "convexHull", "convexDecomposition"],
                    help="mesh collision shape (convexHull lets the robot stand centered + close)")
    ap.add_argument("--verify-cam", action="store_true",
                    help="render a 3rd-person verify_*.png that frames the whole robot at the workspace")
    ap.add_argument("--manipulation-stand", action="store_true",
                    help="place the robot AT the target facing the look-at (task start pose, no navigation)")
    args = ap.parse_args(argv)
    scenarios = json.loads(Path(args.scenarios).read_text())
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios", [])
    m = run_isaac_g1_kitchen_parity_job(
        scenarios=scenarios, out_dir=args.out_dir, kitchen_asset_dir=args.kitchen_asset_dir,
        g1_usd=args.g1_usd, policy_id=args.policy, steps=args.steps, provider=args.provider,
        allow_paid=args.allow_paid, cold=args.cold, image=args.image, max_seconds=args.max_seconds,
        articulated=args.articulated, cheap_collision=args.cheap_collision,
        physics_articulation_drive=args.physics_articulation_drive,
        dynamic_standing_contact_steps=args.dynamic_standing_contact_steps,
        settle_seconds=args.settle_seconds, manipulation_cam=args.manipulation_cam,
        manipulation_look_at=args.manipulation_look_at, render_subframes=args.render_subframes,
        manipulation_reach=args.manipulation_reach, manipulation_reach_arm=args.manipulation_reach_arm,
        fill_light_intensity=args.fill_light_intensity,
        neutral_environment=args.neutral_environment,
        kinematic_arm_pose=args.kinematic_arm_pose,
        collision_approximation=args.collision_approximation, verify_cam=args.verify_cam,
        manipulation_stand=args.manipulation_stand)
    print(json.dumps(m, indent=2, default=str))
    return 0 if m.get("status") in ("completed", "prepared") else 1


if __name__ == "__main__":
    raise SystemExit(main())
