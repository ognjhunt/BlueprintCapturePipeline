"""Worker bootstrap programs for the Isaac G1 kitchen parity lane.

Kept outside the host orchestrator so its provider transaction remains reviewable
and inside source-governance limits.  These strings execute only on the GPU worker.
"""
from __future__ import annotations

# diagnostics-streaming pod bootstrap for the parity runner
BOOTSTRAP = r'''
import os, sys, io, time, json, zipfile, threading, subprocess, urllib.request, pathlib, shutil, signal, stat
OUT="/workspace/out"; BUNDLE="/workspace/bundle"
for d in (OUT, BUNDLE): pathlib.Path(d).mkdir(parents=True, exist_ok=True)
for p in pathlib.Path(OUT).iterdir():
    # tee already holds runner_console.log open; unlinking it detaches the inode and the
    # console then never reaches the output zip (every crash so far collected an empty tail).
    if p.name == "runner_console.log": continue
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except Exception: pass
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
GETB=os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI","")
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")
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
    try: json.dump({"phase":ph, "launch_session_id":SESSION, **k}, open(OUT+"/bootstrap.json","w"))
    except Exception: pass
    putout()
def hb():
    while True:
        time.sleep(25); putout()
def fetch_bytes(url, *, phase, timeout=1800, progress_step=67108864):
    chunks=[]; total=0; next_mark=progress_step
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        content_length=resp.headers.get("Content-Length") or resp.headers.get("content-length")
        mark(phase+"_connected", content_length_bytes=content_length)
        while True:
            chunk=resp.read(4194304)
            if not chunk: break
            chunks.append(chunk); total += len(chunk)
            if total >= next_mark:
                mark(phase+"_progress", bytes_read=total, content_length_bytes=content_length)
                next_mark = total + progress_step
    mark(phase+"_fetched", bytes_read=total)
    return b"".join(chunks)
def safe_extract_zip(data, destination):
    root=pathlib.Path(destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            rel=pathlib.PurePosixPath(member.filename)
            mode=(member.external_attr >> 16) & 0o170000
            if rel.is_absolute() or ".." in rel.parts or stat.S_ISLNK(mode):
                raise RuntimeError("unsafe_zip_member:" + member.filename)
            target=(root / pathlib.Path(*rel.parts)).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise RuntimeError("zip_member_escapes_destination:" + member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink, length=1024*1024)
def runner_timeout_seconds():
    raw=os.environ.get("PARITY_RUNNER_TIMEOUT_SECONDS","")
    if not raw: return 0
    try: return max(1, int(float(raw)))
    except Exception: return 0
threading.Thread(target=hb, daemon=True).start()
try:
    mark("bootstrap_fetching")
    data=fetch_bytes(GETB, phase="bootstrap_fetch", timeout=600, progress_step=16777216)
    safe_extract_zip(data, BUNDLE)
    os.environ["PYTHONPATH"]=BUNDLE + os.pathsep + os.environ.get("PYTHONPATH","")
    mark("bootstrap_extracted", files=sorted(os.listdir(BUNDLE)))
    # kitchen assets are staged separately (large, reused across iterations) and fetched into BUNDLE/kitchen
    KURL=os.environ.get("KITCHEN_BUNDLE_URL","")
    if KURL:
        mark("kitchen_fetching")
        kdata=fetch_bytes(KURL, phase="kitchen_fetch", timeout=1800)
        if os.environ.get("PARITY_SUPERVISED_STARTUP","")=="1":
            pathlib.Path("/workspace/kitchen_assets.zip").write_bytes(kdata)
            mark("kitchen_archive_staged_for_gate", bytes_read=len(kdata))
        else:
            kdir=BUNDLE+"/kitchen"; pathlib.Path(kdir).mkdir(parents=True, exist_ok=True)
            mark("kitchen_extracting", bytes_read=len(kdata))
            safe_extract_zip(kdata, kdir)
            mark("kitchen_extracted", kitchen_files=len(list(pathlib.Path(kdir).rglob("*"))))
    elif os.environ.get("PARITY_SUPERVISED_STARTUP","")=="1":
        raise RuntimeError("supervised_startup_requires_kitchen_bundle")
except Exception as exc:
    mark("bootstrap_failed", error=repr(exc))
    raise
try: subprocess.call(["/isaac-sim/python.sh","-m","pip","install","-q","pillow","google-genai"])  # frame save + Gemini QC deps (best-effort)
except Exception: pass
try: subprocess.call(["bash","-c","command -v ffmpeg >/dev/null 2>&1 || (apt-get update -y >/dev/null 2>&1 && apt-get install -y ffmpeg >/dev/null 2>&1)"])  # mp4 assembly (best-effort)
except Exception: pass
if os.environ.get("PARITY_SERVE_PRODUCTION_WARMUP_BEFORE_READY","")=="1":
    import hashlib
    worker_image_ref=os.environ.get("BLUEPRINT_WORKER_IMAGE_REF","").strip()
    host_image_id=os.environ.get("BLUEPRINT_HOST_IMAGE_ID","").strip()
    gpu_pool_class=os.environ.get("BLUEPRINT_GPU_POOL_CLASS","").strip()
    if "@sha256:" not in worker_image_ref:
        raise RuntimeError("production_worker_image_ref_not_digest_pinned")
    if not host_image_id:
        raise RuntimeError("production_host_image_id_required")
    if gpu_pool_class != "runpod-secure-l40s-preferred-a40-fallback":
        raise RuntimeError("production_gpu_pool_class_invalid")
    gpu_query=subprocess.run(
        ["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
        capture_output=True,text=True,timeout=30,check=False,
    )
    raw_gpu_name=(gpu_query.stdout.splitlines() or [""])[0].strip()
    actual_gpu_model=(
        "NVIDIA L40S" if "L40S" in raw_gpu_name
        else "NVIDIA A40" if "A40" in raw_gpu_name
        else ""
    )
    if gpu_query.returncode != 0 or not actual_gpu_model:
        raise RuntimeError("production_runpod_gpu_model_not_allowed")
    model_cache_root=pathlib.Path(os.environ.get(
        "BLUEPRINT_GROOT_OSCAR_MODEL_CACHE","/models/blueprint-groot-oscar-v1"
    ))
    from blueprint_pipeline.groot_oscar_model_cache import verify_model_cache
    model_verification=verify_model_cache(model_cache_root)
    if model_verification.get("status") != "passed":
        raise RuntimeError("production_model_cache_verification_failed:"+",".join(
            model_verification.get("blockers") or ["unknown_model_cache_failure"]
        ))
    # WBC runtime binaries remain in the foundation; immutable ONNX/config
    # assets are linked from the provider volume only after byte verification.
    gear=model_cache_root/"gear_sonic"
    model_links={
        gear/"model_encoder.onnx":pathlib.Path("/opt/wbc/gear_sonic_deploy/policy/release/model_encoder.onnx"),
        gear/"model_decoder.onnx":pathlib.Path("/opt/wbc/gear_sonic_deploy/policy/release/model_decoder.onnx"),
        gear/"observation_config.yaml":pathlib.Path("/opt/wbc/gear_sonic_deploy/policy/release/observation_config.yaml"),
        gear/"planner_sonic.onnx":pathlib.Path("/opt/wbc/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
    }
    for source,destination in model_links.items():
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() or destination.is_symlink(): destination.unlink()
        destination.symlink_to(source)
    model_manifest_digest=model_verification["model_manifest_digest"]
    pathlib.Path(OUT,"production_model_cache_manifest.json").write_text(
        json.dumps({**model_verification,"worker_image_ref":worker_image_ref},indent=2),
        encoding="utf-8",
    )
    pathlib.Path(OUT,"production_host_boot_evidence.json").write_text(json.dumps({
        "schema_version":"production_gpu_host_boot_evidence.v1",
        "host_image_id":host_image_id,"actual_gpu_model":actual_gpu_model,
        "checks":{"host_image_booted":True,"nvidia_driver_ready":True,
                  "container_runtime_ready":True},
    },indent=2),encoding="utf-8")
    pathlib.Path(OUT,"production_cache_evidence.json").write_text(json.dumps({
        "schema_version":"production_gpu_cache_evidence.v1",
        "worker_image_ref":worker_image_ref,"model_manifest_digest":model_manifest_digest,
        "checks":{"worker_image_cached":True,"models_cached_offline":True},
    },indent=2),encoding="utf-8")
    mark("production_worker_local_evidence_ready", actual_gpu_model=actual_gpu_model,
         model_manifest_digest=model_manifest_digest)
if os.environ.get("PARITY_SUPERVISED_STARTUP","")=="1":
    image_digest=os.environ.get("BLUEPRINT_WORKER_IMAGE_DIGEST","")
    gate_rows=[]; gate_blockers=[]
    def run_process_group_bounded(command, *, timeout, terminate_grace=5.0):
        process=subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ), start_new_session=True,
        )
        try:
            stdout,stderr=process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGTERM)
            except Exception: pass
            grace_deadline=time.monotonic()+max(0.0,terminate_grace)
            try:
                process.wait(timeout=max(0.0,terminate_grace))
            except subprocess.TimeoutExpired: pass
            remaining=grace_deadline-time.monotonic()
            if remaining>0: time.sleep(remaining)
            # Kill the group regardless of whether its leader already exited:
            # a descendant may close the pipes, ignore TERM, and keep running.
            try: os.killpg(process.pid, signal.SIGKILL)
            except Exception: pass
            try: stdout,stderr=process.communicate(timeout=30)
            except Exception: stdout,stderr="","process_group_kill_failed"
            return subprocess.CompletedProcess(
                command,124,stdout or "",(stderr or "")+"\nprocess_group_timeout"
            )
    def run_startup_gate(gate_id, command, result_path, passed_status):
        mark(gate_id+"_starting")
        started=time.monotonic()
        try:
            completed=run_process_group_bounded(command, timeout=1800)
            rc=completed.returncode
            stdout=completed.stdout[-4000:]; stderr=completed.stderr[-4000:]
        except Exception as exc:
            rc=124; stdout=""; stderr=repr(exc)
        try: payload=json.loads(pathlib.Path(result_path).read_text(encoding="utf-8"))
        except Exception as exc: payload={"status":"missing", "blockers":[gate_id+"_result_missing"], "error":repr(exc)}
        payload["launch_session_id"]=SESSION
        payload["image_digest"]=image_digest
        pathlib.Path(result_path).write_text(json.dumps(payload,indent=2),encoding="utf-8")
        passed=rc==0 and payload.get("status")==passed_status
        row={"gate_id":gate_id, "status":"passed" if passed else "blocked",
             "exit_code":rc, "elapsed_seconds":round(time.monotonic()-started,1),
             "result_path":str(result_path), "stdout_tail":stdout, "stderr_tail":stderr,
             "blockers":[] if passed else list(payload.get("blockers") or [gate_id+"_failed"])}
        gate_rows.append(row)
        if not passed: gate_blockers.extend(row["blockers"])
        mark(gate_id+"_passed" if passed else gate_id+"_blocked", blockers=row["blockers"])
        return passed
    kitchen_gate_dir=pathlib.Path("/workspace/kitchen_asset_gate")
    kitchen_gate_path=kitchen_gate_dir/"kitchen_asset_startup_gate.json"
    kitchen_passed=run_startup_gate(
        "kitchen_asset_startup_gate",
        ["/isaac-sim/python.sh","-m","blueprint_pipeline.kitchen_asset_startup_gate",
         "--expected-inventory",BUNDLE+"/kitchen_asset_inventory_checksums.json",
         "--out-dir",str(kitchen_gate_dir),"--archive","/workspace/kitchen_assets.zip"],
        kitchen_gate_path,
        "completed",
    )
    if kitchen_gate_path.is_file():
        evidence_dir=pathlib.Path(OUT,"kitchen_asset_gate"); evidence_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(kitchen_gate_path,evidence_dir/"kitchen_asset_startup_gate.json")
    if kitchen_passed:
        gate_payload=json.loads(kitchen_gate_path.read_text(encoding="utf-8"))
        kdir=pathlib.Path(BUNDLE,"kitchen")
        if kdir.exists(): shutil.rmtree(kdir)
        shutil.move(str(gate_payload["materialized_tree"]),str(kdir))
        mark("kitchen_asset_gate_materialized", kitchen_files=gate_payload.get("verified_file_count"),
             kitchen_bytes=gate_payload.get("verified_total_bytes"))
        preflight_path=pathlib.Path(OUT,"isaac_worker_runtime_preflight.json")
        fast_passed=run_startup_gate(
            "fast_startup_canary",
            ["/isaac-sim/python.sh","-m","blueprint_pipeline.isaac_worker_runtime_preflight",
             "--output",str(preflight_path),"--require-nvidia-smi","--require-rtx-render","--smoke-steps","3"],
            preflight_path,
            "passed",
        )
        if fast_passed:
            review_dir=pathlib.Path(OUT,"review_renderer_canary")
            review_path=review_dir/"isaac_review_renderer_canary.json"
            run_startup_gate(
                "review_renderer_canary",
                ["/isaac-sim/python.sh","-m","blueprint_pipeline.isaac_review_renderer_canary",
                 "--output-dir",str(review_dir),"--launch-session-id",SESSION,
                 "--image-digest",image_digest,"--orientation","landscape"],
                review_path,
                "passed",
            )
    summary={"schema_version":"isaac_supervised_startup_gates.v1",
             "status":"passed" if not gate_blockers else "blocked",
             "blockers":sorted(set(gate_blockers)), "launch_session_id":SESSION,
             "image_digest":image_digest, "gates":gate_rows,
             "claim_boundary":{"startup_and_asset_proof_only":True,
                               "kitchen_scene_load_proven":False,
                               "policy_execution_proven":False,
                               "proves_task_success":False}}
    pathlib.Path(OUT,"supervised_startup_gates.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    mark("startup_gates_passed" if not gate_blockers else "startup_gates_blocked",
         blockers=summary["blockers"])
    if gate_blockers:
        while True:
            time.sleep(30); putout()
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
if os.environ.get("PARITY_GROOT_POLICY_COMMAND",""): cmd += ["--groot-policy-command", os.environ["PARITY_GROOT_POLICY_COMMAND"]]
if os.environ.get("PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS",""): cmd += ["--groot-policy-command-timeout-seconds", os.environ["PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"]]
if os.environ.get("PARITY_GROOT_POLICY_INITIAL_FRAME",""): cmd += ["--groot-policy-initial-frame", os.environ["PARITY_GROOT_POLICY_INITIAL_FRAME"]]
if os.environ.get("PARITY_DYNAMIC_EPISODE_TERMINATION","")=="1": cmd.append("--dynamic-episode-termination")
elif os.environ.get("PARITY_DYNAMIC_EPISODE_TERMINATION","")=="0": cmd.append("--no-dynamic-episode-termination")
if os.environ.get("PARITY_EPISODE_MAX_STEPS",""): cmd += ["--episode-max-steps", os.environ["PARITY_EPISODE_MAX_STEPS"]]
if os.environ.get("PARITY_DYNAMIC_EPISODE_CHECK_EVERY",""): cmd += ["--dynamic-episode-check-every", os.environ["PARITY_DYNAMIC_EPISODE_CHECK_EVERY"]]
if os.environ.get("PARITY_CAPTURE_EVERY",""): cmd += ["--capture-every", os.environ["PARITY_CAPTURE_EVERY"]]
if os.environ.get("PARITY_KEEP_OBJECTS",""): cmd += ["--keep-objects", os.environ["PARITY_KEEP_OBJECTS"]]
if os.environ.get("PARITY_NO_PROBE","")=="1": cmd.append("--no-collision-probe")
if os.environ.get("PARITY_CHEAP_COLLISION","")=="1": cmd.append("--cheap-collision")
if os.environ.get("PARITY_ARTICULATED","")=="1": cmd.append("--articulated")
if os.environ.get("PARITY_PHYSICS_ARTICULATION_DRIVE","")=="1": cmd.append("--physics-articulation-drive")
if os.environ.get("PARITY_DYNAMIC_STANDING_CONTACT_STEPS",""): cmd += ["--dynamic-standing-contact-steps", os.environ["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"]]
if os.environ.get("PARITY_MANIPULATION_CAM","")=="1": cmd.append("--manipulation-cam")
if os.environ.get("PARITY_MANIPULATION_LOOK_AT",""):
    # Keep a negative leading X coordinate attached to the option.  Passing it as the next argv
    # token makes argparse interpret values such as ``-1.59,1.47,1.24`` as another option and the
    # paid worker exits before Isaac starts.
    cmd += ["--manipulation-look-at=" + os.environ["PARITY_MANIPULATION_LOOK_AT"]]
if os.environ.get("PARITY_RENDER_SUBFRAMES",""): cmd += ["--render-subframes", os.environ["PARITY_RENDER_SUBFRAMES"]]
if os.environ.get("PARITY_NO_SOFTWARE_DENOISE","")=="1": cmd.append("--no-software-denoise")
if os.environ.get("PARITY_MANIPULATION_REACH","")=="1": cmd.append("--manipulation-reach")
if os.environ.get("PARITY_MANIPULATION_REACH_ARM",""): cmd += ["--manipulation-reach-arm", os.environ["PARITY_MANIPULATION_REACH_ARM"]]
if os.environ.get("PARITY_FILL_LIGHT_INTENSITY",""): cmd += ["--fill-light-intensity", os.environ["PARITY_FILL_LIGHT_INTENSITY"]]
if os.environ.get("PARITY_NEUTRAL_ENVIRONMENT","")=="1": cmd.append("--neutral-environment")
if os.environ.get("PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE","")=="1": cmd.append("--robot-review-material-override")
if os.environ.get("PARITY_ROBOT_REVIEW_MATERIAL_MODE",""): cmd += ["--robot-review-material-mode", os.environ["PARITY_ROBOT_REVIEW_MATERIAL_MODE"]]
if os.environ.get("PARITY_COLLISION_APPROXIMATION",""): cmd += ["--collision-approximation", os.environ["PARITY_COLLISION_APPROXIMATION"]]
if os.environ.get("PARITY_VERIFY_CAM","")=="1": cmd.append("--verify-cam")
if os.environ.get("PARITY_MANIPULATION_STAND","")=="1": cmd.append("--manipulation-stand")
if os.environ.get("PARITY_NO_PLACEMENT_TOPDOWN_CAPTURE","")=="1": cmd.append("--no-placement-topdown-capture")
if os.environ.get("PARITY_KINEMATIC_ARM_POSE","")=="1": cmd.append("--kinematic-arm-pose")
if os.environ.get("PARITY_RENDER_NOISE_AUDIT","")=="1":
    cmd.append("--render-noise-audit")  # variant-matrix render-quality audit instead of the scenario eval
    if os.environ.get("PARITY_AUDIT_HIGH_SPP",""): cmd += ["--audit-high-spp", os.environ["PARITY_AUDIT_HIGH_SPP"]]
    if os.environ.get("PARITY_AUDIT_WARMUP_FRAMES",""): cmd += ["--audit-warmup-frames", os.environ["PARITY_AUDIT_WARMUP_FRAMES"]]
    if os.environ.get("PARITY_AUDIT_BOOST_LIGHT_INTENSITY",""): cmd += ["--audit-boost-light-intensity", os.environ["PARITY_AUDIT_BOOST_LIGHT_INTENSITY"]]
if os.environ.get("PARITY_SERVE","")=="1":
    cmd.append("--serve")  # warm mode: boot Isaac + load scene ONCE, then serve jobs from the inbox env
    if os.environ.get("PARITY_SERVE_IDLE_TIMEOUT",""): cmd += ["--serve-idle-timeout", os.environ["PARITY_SERVE_IDLE_TIMEOUT"]]
    if os.environ.get("PARITY_SERVE_MAX_JOBS",""): cmd += ["--serve-max-jobs", os.environ["PARITY_SERVE_MAX_JOBS"]]
    if os.environ.get("PARITY_SERVE_PRODUCTION_WARMUP_BEFORE_READY","")=="1": cmd.append("--serve-production-warmup-before-ready")
mark("runner_starting", cmd=cmd)
timeout=runner_timeout_seconds()
started=time.monotonic()
proc=subprocess.Popen(cmd, start_new_session=True)
try:
    rc=proc.wait(timeout=timeout if timeout > 0 else None)
    mark("runner_done", rc=rc, elapsed_seconds=round(time.monotonic()-started,1),
         timeout_seconds=timeout)
except subprocess.TimeoutExpired:
    try: os.killpg(proc.pid, signal.SIGTERM)
    except Exception: pass
    grace_deadline=time.monotonic()+30
    try:
        rc=proc.wait(timeout=30)
    except Exception:
        rc=None
    # The group leader can exit on SIGTERM while a descendant ignores it.
    # Honor the full grace window, then kill the group regardless of whether
    # the leader was already reaped.
    remaining=grace_deadline-time.monotonic()
    if remaining > 0: time.sleep(remaining)
    try: os.killpg(proc.pid, signal.SIGKILL)
    except Exception: pass
    try: rc=proc.wait(timeout=10)
    except Exception: pass
    mark("runner_timeout", rc=rc, timeout_seconds=timeout,
         elapsed_seconds=round(time.monotonic()-started,1), cmd=cmd)
# Keep the container process alive after runner completion so RunPod does not restart it and
# clobber the final output object before the parent collector observes runner_done.
while True:
    time.sleep(30); putout()
'''

IMAGE_STARTUP_CANARY_BOOTSTRAP = r'''
import os, io, json, time, zipfile, pathlib, urllib.request, shutil, sys, subprocess, signal, threading
from datetime import datetime, timezone

OUT="/workspace/out"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
for p in pathlib.Path(OUT).iterdir():
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except Exception:
        pass
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
GETB=os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI","")
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")

def putout():
    try:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in pathlib.Path(OUT).rglob("*"):
                if p.is_file():
                    try:
                        z.write(p, p.relative_to(OUT).as_posix())
                    except Exception:
                        pass
        req=urllib.request.Request(PUT, data=buf.getvalue(), method="PUT", headers={"Content-Type":"application/zip"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass

def mark(phase, **extra):
    payload={"phase":phase, "launch_session_id":SESSION, **extra}
    pathlib.Path(OUT, "bootstrap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    putout()

CANARY_BUNDLE="/workspace/canary_bundle"
pathlib.Path(CANARY_BUNDLE).mkdir(parents=True, exist_ok=True)
bundle_bytes=0
bundle_files=[]
bundle_error=None
try:
    if not GETB:
        raise RuntimeError("canary_bundle_url_missing")
    with urllib.request.urlopen(GETB, timeout=600) as response:
        bundle_data=response.read(134217728)
    bundle_bytes=len(bundle_data)
    zipfile.ZipFile(io.BytesIO(bundle_data)).extractall(CANARY_BUNDLE)
    bundle_files=sorted(
        p.relative_to(CANARY_BUNDLE).as_posix()
        for p in pathlib.Path(CANARY_BUNDLE).rglob("*") if p.is_file()
    )
    os.environ["PYTHONPATH"]=CANARY_BUNDLE + os.pathsep + os.environ.get("PYTHONPATH","")
except Exception as exc:
    bundle_error=repr(exc)
mark("runtime_preflight_starting", image_startup_canary=True,
     bundle_bytes=bundle_bytes, bundle_file_count=len(bundle_files), bundle_error=bundle_error)
preflight_path=pathlib.Path(OUT, "isaac_worker_runtime_preflight.json")
preflight_cmd=[
    "/isaac-sim/python.sh", "-m", "blueprint_pipeline.isaac_worker_runtime_preflight",
    "--output", str(preflight_path), "--require-nvidia-smi", "--require-rtx-render",
    "--smoke-steps", "3",
]
started=time.monotonic()
preflight_deadline_done=threading.Event()
preflight_process_holder={}
def enforce_preflight_deadline():
    if preflight_deadline_done.wait(930):
        return
    process=preflight_process_holder.get("process")
    if process is not None:
        try: os.killpg(process.pid, signal.SIGKILL)
        except Exception: pass
    timeout_result={
        "schema_version":"isaac_g1_parity_image_startup_canary.v2",
        "status":"blocked",
        "blockers":["isaac_runtime_preflight_process_group_timeout"],
        "image_startup_canary":True,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "launch_session_id":SESSION,
        "runtime_preflight_exit_code":124,
        "runtime_preflight_elapsed_seconds":round(time.monotonic()-started,1),
        "claim_boundary":"The independent canary watchdog expired before the Isaac RTX preflight returned. This is a bounded startup failure, not simulator, task, policy, or robot-readiness proof.",
    }
    for evidence_path in (
        preflight_path,
        pathlib.Path(OUT, "isaac_g1_kitchen_parity_result.json"),
        pathlib.Path(OUT, "isaac_g1_parity_image_startup_canary.json"),
    ):
        try: evidence_path.write_text(json.dumps(timeout_result, indent=2), encoding="utf-8")
        except Exception: pass
    try:
        mark("runner_done", rc=124, image_startup_canary=True, runtime_preflight_status="blocked",
             watchdog_timeout=True)
    except Exception: pass
    # RunPod restarts a pod when its container command exits. Keep this
    # watchdog thread alive and continue publishing the terminal blocked
    # artifact so one timed-out canary cannot restart itself indefinitely.
    while True:
        time.sleep(30)
        putout()
threading.Thread(target=enforce_preflight_deadline, daemon=True).start()
try:
    if bundle_error:
        raise RuntimeError("canary_bundle_unavailable:" + bundle_error)
    preflight_process=subprocess.Popen(
        preflight_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ), start_new_session=True,
    )
    preflight_process_holder["process"]=preflight_process
    try:
        preflight_stdout,preflight_stderr=preflight_process.communicate(timeout=900)
        preflight_rc=preflight_process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(preflight_process.pid, signal.SIGKILL)
        preflight_stdout,preflight_stderr=preflight_process.communicate(timeout=30)
        preflight_rc=124
        preflight_stderr=(preflight_stderr or "") + "\nisaac_runtime_preflight_process_group_timeout"
    preflight_stdout=preflight_stdout[-4000:]
    preflight_stderr=preflight_stderr[-4000:]
except Exception as exc:
    preflight_rc=124
    preflight_stdout=""
    preflight_stderr=repr(exc)
finally:
    preflight_deadline_done.set()
try:
    preflight=json.loads(preflight_path.read_text(encoding="utf-8"))
except Exception as exc:
    preflight={"status":"blocked", "blockers":["isaac_runtime_preflight_result_missing"], "error":repr(exc)}
preflight_passed=preflight_rc == 0 and preflight.get("status") == "passed"
workspace_disk=shutil.disk_usage("/workspace")
canary={
    "schema_version": "isaac_g1_parity_image_startup_canary.v2",
    "status": "completed" if preflight_passed else "blocked",
    "blockers": [] if preflight_passed else list(preflight.get("blockers") or ["isaac_runtime_preflight_failed"]),
    "image_startup_canary": True,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "launch_session_id": SESSION,
    "python_executable": sys.executable,
    "python3_path": shutil.which("python3"),
    "isaac_python_path": "/isaac-sim/python.sh" if pathlib.Path("/isaac-sim/python.sh").exists() else None,
    "blueprint_worker_image_family": os.environ.get("BLUEPRINT_WORKER_IMAGE_FAMILY"),
    "simulator_framework": os.environ.get("BLUEPRINT_SIMULATOR_FRAMEWORK"),
    "image_contract_mode": "split_isaac_carrier_plus_signed_blueprint_bundle",
    "signed_bundle_bytes": bundle_bytes,
    "signed_bundle_file_count": len(bundle_files),
    "signed_bundle_error": bundle_error,
    "runtime_preflight_command": preflight_cmd,
    "runtime_preflight_exit_code": preflight_rc,
    "runtime_preflight_elapsed_seconds": round(time.monotonic()-started, 1),
    "runtime_preflight_stdout_tail": preflight_stdout,
    "runtime_preflight_stderr_tail": preflight_stderr,
    "runtime_preflight": preflight,
    "workspace_disk": {
        "total_bytes": workspace_disk.total,
        "used_bytes": workspace_disk.used,
        "free_bytes": workspace_disk.free,
    },
    "claim_boundary": (
        "This canary proves only that the selected split Isaac carrier plus signed Blueprint "
        "bundle reached user command execution, uploaded a provider output artifact, and ran "
        "the explicit CUDA/Isaac/64x64 RTX runtime preflight. It does not prove "
        "kitchen scene loading, policy execution, task success, WAM quality, or robot readiness."
    ),
}
pathlib.Path(OUT, "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(canary, indent=2), encoding="utf-8")
pathlib.Path(OUT, "isaac_g1_parity_image_startup_canary.json").write_text(json.dumps(canary, indent=2), encoding="utf-8")
mark("runner_done", rc=preflight_rc, image_startup_canary=True,
     runtime_preflight_status=preflight.get("status"))
while True:
    time.sleep(30)
    putout()
'''

_EARLY_MARKER = r'''
import os, io, json, zipfile, pathlib, urllib.request
OUT="/workspace/out"; pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")
json.dump({"phase":"container_bash_started","launch_session_id":SESSION}, open(OUT+"/bootstrap.json","w"))
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
buf=io.BytesIO()
with zipfile.ZipFile(buf,"w") as z: z.write(OUT+"/bootstrap.json","bootstrap.json")
try: urllib.request.urlopen(urllib.request.Request(PUT,data=buf.getvalue(),method="PUT",headers={"Content-Type":"application/zip"}),timeout=60).read()
except Exception: pass
'''


def docker_start_cmd(*, image_startup_canary: bool = False) -> list[str]:
    worker_script = IMAGE_STARTUP_CANARY_BOOTSTRAP if image_startup_canary else BOOTSTRAP
    worker_script_name = "parity_image_startup_canary.py" if image_startup_canary else "boot.py"
    worker_python_cmd = (
        "if command -v python3 >/dev/null 2>&1; then "
        f"python3 /workspace/{worker_script_name}; "
        "elif command -v python >/dev/null 2>&1; then "
        f"python /workspace/{worker_script_name}; "
        "else "
        f"/isaac-sim/python.sh /workspace/{worker_script_name}; "
        "fi"
        if image_startup_canary
        else f"/isaac-sim/python.sh /workspace/{worker_script_name}"
    )
    script = (
        "set +e\n"
        "set -o pipefail\n"
        "mkdir -p /workspace/out\n"
        "cat > /workspace/early.py <<'EARLYEOF'\n" + _EARLY_MARKER + "\nEARLYEOF\n"
        "(python3 /workspace/early.py 2>/dev/null || /isaac-sim/python.sh /workspace/early.py 2>/dev/null) || true\n"
        f"cat > /workspace/{worker_script_name} <<'PYEOF'\n" + worker_script + "\nPYEOF\n"
        f"{worker_python_cmd} 2>&1 | tee /workspace/out/runner_console.log\n"
    )
    return ["-lc", script]
