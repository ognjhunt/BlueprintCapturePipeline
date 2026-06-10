# First GPU E2E Runbook

Use this runbook for the first sample-video pass that intentionally crosses from
local capture/package automation into owner GPU simulator execution.

This is the current path:

```text
BlueprintCapture raw bundle
  -> BlueprintCapture bridge
  -> BlueprintCapturePipeline current lanes
  -> Blueprint-WebApp robot-eval request
  -> BlueprintCapturePipeline live intake/control plane
  -> owner GPU simulator command
  -> proof ingestion and closure audit
```

The first successful GPU pass proves only the accepted simulator command and its
evidence artifacts. It does not prove robot readiness, safety/contact validity,
live policy success, customer delivery, or public-claim upgrades.

## Runtime Choice

Use a GPU VM or pod for the first run. RunPod, Vast, NVIDIA Brev, AWS, Azure, or
GCP can all work if the instance gives you Linux, Docker, NVIDIA Container
Toolkit, persistent storage, outbound asset access, and an RTX-capable GPU.

Do not treat NVIDIA NIMs as the primary simulator runtime. NIMs are useful later
for model inference services, but they do not replace Isaac Sim or Isaac Lab
execution.

For the first proof run:

- Prefer `isaac_sim` when the goal is rich USD/OpenUSD scene load, robot spawn,
  sensor/action smoke, and proof artifact collection.
- Use `isaac_lab_arena` after the scene, robot profile, task binding, and Arena
  packet are ready for scalable scenario evaluation.
- Keep MuJoCo, PyBullet, or Newton as fast proxy checks unless the owner accepts
  them as the selected simulator backend for a specific proof.
- Pick an RTX/RT-core GPU. Isaac Sim requires RT-core capable GPUs; A100/H100 are
  not the right first target even though they are strong training/inference GPUs.
  L40S 48GB or RTX 6000 Ada 48GB are the preferred first targets. RTX 4090 24GB
  is acceptable for a cheaper smoke if the scene is small.

Reference docs:

- NVIDIA NIM: https://developer.nvidia.com/nim
- Isaac Sim container install: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html
- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
- Isaac Lab: https://developer.nvidia.com/isaac/lab
- RunPod pricing: https://www.runpod.io/pricing

## Inputs To Lock Before Running

Set these values in your shell or run sheet:

```bash
export CAPTURE_ROOT="/path/to/local/bucket/scenes/<scene_id>/captures/<capture_id>"
export WEBAPP_SITE_SLUG="<site-slug>"
export ROBOT_EVAL_JOB_ID="<job-id-if-known>"
export GPU_PROOF_DIR="$CAPTURE_ROOT/pipeline/simulation_automation/owner_gpu_proof"
```

If you are not sure which local staged capture is the sample-video candidate,
scan the current artifacts first:

```bash
blueprint-audit-first-gpu-candidates \
  --search-root output \
  --output output/first_gpu_candidate_audit_manifest.json
```

This audit looks for `raw/capture_upload_complete.json` capture roots, checks
for a raw walkthrough video, then runs the first-GPU readiness audit for each
candidate. It does not run providers, WebApp, simulators, or GPU provisioning.
Do not spend GPU time until the intended sample capture appears in this report
with the expected raw video and without local/WebApp/gate blockers.

If the sample is still just a local video file, audit the file before staging
it. The first World Labs clip should be short and small enough to keep the
first pass cheap and debuggable; the default preflight limit is 30 seconds and
100 MB.

```bash
blueprint-audit-first-gpu-sample-video \
  --source-video /path/to/collected-sample.mp4 \
  --output output/first_gpu_sample_video_preflight_manifest.json
```

This manifest separates `ready_for_capture_staging` from
`ready_for_worldlabs_first_clip`. It checks local file existence, supported
video suffix, size, and ffprobe duration metadata when available. It does not
prove privacy clearance, scene geometry, WebApp upstream truth, simulator
execution, or robot readiness.

When the sample preflight is ready, stage it into the supported capture-tree
layout:

```bash
blueprint-stage-first-gpu-sample-video \
  --source-video /path/to/collected-sample.mp4 \
  --storage-root output/first-gpu-sample-storage \
  --bucket local-blueprint \
  --scene-id first-gpu-sample-site \
  --capture-id first-gpu-sample-capture \
  --require-source-video-preflight \
  --workflow-name "First GPU sample walkthrough" \
  --task-step "load captured scene" \
  --task-step "spawn robot at proposed start pose" \
  --task-step "attempt the selected task trace" \
  --scene-asset /path/to/materialized-scene.obj \
  --run-simulation-automation
```

Only pass `--site-submission-id`, `--request-id`, `--buyer-request-id`, and
`--capture-job-id` if those values came from the real WebApp/Capture job path.
Leaving them unset is correct for a local staging rehearsal; the readiness audit
will keep the WebApp leg blocked until real upstream truth is staged.

If you have the real upstream IDs but want a local dry run before submitting
from the WebApp, add `--stage-local-webapp-rehearsal-request`. This writes a
local `robot_eval_job_request.v1` queue envelope plus
`pipeline/live_pipeline_staged_inputs.json`, but marks both as
`local_first_gpu_rehearsal_request`. The first-GPU readiness audit still blocks
that evidence by default; pass `--allow-local-webapp-rehearsal` only for a
local rehearsal and do not treat it as WebApp forwarding proof.

The staging command preserves the video as a `pre_screen_video` bundle. It can
write local simulation automation artifacts, but it does not infer camera pose,
intrinsics, depth, robot policy packages, or WebApp IDs from the video. If the
result reports `gpu_handoff_blockers` such as `spawn_validation_blocked`, fix the
scene/spawn inputs before renting GPU time.

Use `--scene-asset` only for a local materialized scene asset that genuinely
belongs to the captured sample, such as a World Labs, SimReady, OpenUSD, glTF,
OBJ, PLY, URDF, or MJCF export. A valid scene asset can clear local scene-bounds
and spawn-sanity blockers, but it still does not prove owner GPU simulator
execution.

The raw bundle must include:

- `raw/manifest.json`
- `raw/capture_context.json`
- `raw/capture_upload_complete.json`
- a real walkthrough video such as `raw/walkthrough.mov` or
  `raw/walkthrough.mp4`
- truthful `capture_capabilities`
- `requested_outputs` containing `robot_eval_dataset` and `task_evaluation_run`
- rights/privacy metadata that allows the intended downstream evaluation

For the WebApp leg, the request must also carry real upstream IDs:

- `site_submission_id`
- `request_id` or `owner_system.request_id`
- `buyer_request_id`
- `capture_job_id`

Placeholders, capture-derived IDs, and `/synced-artifacts/sites/<slug>` paths do
not prove WebApp upstream truth.

## Phase 0.5: Cross-Repo Contract Audit

Before spending GPU time, run the cross-repo audit from
`BlueprintCapturePipeline`:

```bash
blueprint-audit-first-gpu-cross-repo-readiness \
  --capture-repo /Users/nijelhunt_1/workspace/BlueprintCapture \
  --webapp-repo /Users/nijelhunt_1/workspace/Blueprint-WebApp \
  --capture-root "$CAPTURE_ROOT" \
  --webapp-site-slug "$WEBAPP_SITE_SLUG" \
  --simulator isaac_sim \
  --provisioner runpod \
  --simulator-command "$OWNER_SIMULATOR_COMMAND" \
  --simulator-command-location remote
```

Without `--capture-root`, the audit can prove only source-contract coverage for
Capture -> Pipeline -> WebApp -> Pipeline. With `--capture-root`, it wraps the
authoritative first-GPU readiness audit and reads
`pipeline/first_gpu_e2e_run_packet/first_gpu_launch_order.json`,
`first_gpu_blocker_resolution.json`, `first_gpu_webapp_handoff.json`,
`first_gpu_scene_asset_acquisition.json`, `gpu_vm_runtime_preflight_plan.json`,
and `gpu_vm_sync_manifest.json`. It should stay blocked until the actual sample
capture, staged WebApp request, forwarding env, WebApp handoff packet,
scene-acquisition evidence, scene/spawn inputs, owner command, GPU gates, launch
order, VM sync, and VM preflight plan are all ready.

Read `gpu_spend_decision` before provisioning any RunPod or equivalent GPU VM.
When it says `do_not_rent_gpu_yet`, the manifest also records the forbidden
actions, including not allocating GPU time and not running `gpu_vm_commands.sh`.
Then read `first_gpu_external_input_packet`: it condenses the remaining
external inputs into ordered categories, names required IDs, env vars, scene
artifacts, owner GPU command inputs, and VM checks, and redacts secret values
while still naming secret keys such as `WORLDLABS_API_KEY` and
`ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`. When the cross-repo audit writes an
output manifest, it also writes `first_gpu_external_input_packet.md` beside it
as the operator-readable checklist.
Then read `first_gpu_operator_actions` for the exact ordered packet actions and
`remediation_plan.categories` for the cross-repo fix lanes such as
`webapp_upstream_truth`, `webapp_forwarding_env`, `webapp_staged_request`,
`webapp_handoff_packet`, `scene_asset_acquisition`, `scene_spawn_preflight`,
`pipeline_gpu_handoff`, `first_gpu_run_packet`, `gpu_vm_sync`,
`gpu_vm_runtime_preflight`, `owner_gpu_command`, and `owner_gpu_gate`, with the
required evidence and safe command for each lane.

For a local request-shape rehearsal that was written with
`--stage-local-webapp-rehearsal-request`, add
`--allow-local-webapp-rehearsal`. Do not use that flag for the real
WebApp-forwarded E2E gate. The cross-repo audit still treats local rehearsal as
insufficient for the full Capture -> Pipeline -> WebApp -> Pipeline spend
decision and keeps `gpu_spend_decision.gpu_rental_recommended_now=false` until
the staged request is no longer marked `local_first_gpu_rehearsal_request`.

## Phase 1: Local Capture And Pipeline Preflight

Run from `BlueprintCapturePipeline`:

```bash
blueprint-preflight-capture \
  --capture-root "$CAPTURE_ROOT" \
  --output "$CAPTURE_ROOT/pipeline/preflight_capture_report.json"
```

Stop if the report is `blocked` or lists missing raw inputs.

Then run the current local path:

```bash
blueprint-run-e2e \
  --capture-root "$CAPTURE_ROOT" \
  --provider openai \
  --pipeline-lane current \
  --run-evaluation-prep \
  --evaluation-prep-provider manual
```

This should create or refresh qualification, evaluation-prep, and simulation
automation artifacts. It does not run GPU simulation.

Run the simulation automation lane explicitly so the GPU handoff packet is fresh:

```bash
blueprint-run-simulation-automation \
  --capture-root "$CAPTURE_ROOT"
```

Expected local artifacts:

- `pipeline/simulation_automation/gpu_handoff_packet.json`
- `pipeline/simulation_automation/gpu_owner_system_proof_schema.json`
- `pipeline/simulation_automation/gpu_run_checklist.md`
- `pipeline/simulation_automation/owner_gpu_simulator_execution_blocked_manifest.json`
- `pipeline/simulation_automation/simulator_engine_plugin_registry.json`

Expected blocker before GPU:

```text
owner_gpu_simulator_execution_not_run
```

That blocker is correct. Do not bypass it with CPU-only artifacts.

If `gpu_handoff_packet.json` also lists blockers such as
`spawn_validation_blocked`, the capture is not yet ready for a useful owner GPU
attempt. Add or repair the scene assets, task anchors, spawn pose evidence, or
manual review artifacts that the handoff packet names, then rerun simulation
automation before provisioning a GPU.

Use `gpu_handoff_packet.json.pre_gpu_blocker_details` as the operator fix list.
For a single-video rehearsal, the likely hard blockers are
`missing_local_scene_asset`, `missing_scene_frame_estimate`, and
`scene_bounds_missing_or_invalid`; those mean Pipeline needs materialized scene
geometry or finite scene bounds before owner GPU simulator execution can produce
useful proof.

Then run the first-GPU readiness audit. Before the WebApp request is staged and
the GPU command is configured, it should report those missing inputs explicitly:

```bash
blueprint-audit-first-gpu-e2e-readiness \
  --capture-root "$CAPTURE_ROOT" \
  --webapp-site-slug "$WEBAPP_SITE_SLUG" \
  --simulator isaac_sim \
  --provisioner runpod \
  --owner-command "$OWNER_SIMULATOR_COMMAND" \
  --owner-command-location remote
```

To create the concrete operator packet for the local shell and GPU VM, generate
the first-GPU run packet:

```bash
blueprint-build-first-gpu-run-packet \
  --capture-root "$CAPTURE_ROOT" \
  --webapp-site-slug "$WEBAPP_SITE_SLUG" \
  --simulator isaac_sim \
  --provisioner runpod
```

This writes `pipeline/first_gpu_e2e_run_packet/` with a readiness manifest,
`first_gpu_env.example`, `local_preflight_commands.sh`,
`webapp_upstream_truth_verification_commands.sh`, `gpu_vm_commands.sh`,
`gpu_vm_runtime_preflight.sh`, `gpu_vm_runtime_preflight_plan.json` and `.md`,
`first_gpu_simulator_path_matrix.json` and `.md`,
`first_gpu_launch_order.json` and `.md`,
`owner_command_contract.md`, `gpu_provider_bootstrap.md`, and
`gpu_provider_bootstrap.json`, plus `first_gpu_blocker_resolution.json` and
`.md`, `first_gpu_scene_asset_acquisition.json` and `.md`,
`first_gpu_webapp_handoff.json` and `.md`, plus `gpu_vm_sync_manifest.json` and
`.md`. The provider bootstrap files record the
selected RunPod/equivalent GPU VM path, Isaac GPU constraints, NIM boundary,
mount/sync requirements, success criteria, and hard stops. The simulator matrix
files distinguish the selected first-GPU backend from follow-up Arena/policy,
MuJoCo/PyBullet preflight, Newton, and NIM inference-service roles. The launch
order files mark which phases may run now and forbid GPU commands until WebApp,
scene, sync, VM preflight, owner-command, and simulator gates are ready. Once
those pre-run gates are ready, `owner_gpu_simulator_proof` may run even though
`post_gpu_readiness_audit` remains pending until proof files exist. The
blocker-resolution files turn the current audit blockers into an ordered fix
list for source video, WebApp upstream truth, forwarding env, staged WebApp
request, scene/spawn preflight, owner command, and explicit GPU gates. The JSON
also includes top-level `actions`, `action_count`, and `blocked_action_count`
so operators and scripts can consume the fix list without inferring it from
category detail. Scene and GPU-handoff actions include `blocker_details` copied
from `gpu_handoff_packet.json.pre_gpu_blocker_details`, so hard preflight inputs
such as missing scene assets, missing scene-frame estimates, and invalid scene
bounds stay visible in the operator packet. WebApp upstream actions carry
field-level required IDs and accepted evidence sources, including raw/capture
manifests, Pipeline handoff files, and staged `robot_eval_job_request.v1`
owner-system or site-package fields. Owner-command actions carry the expected
`blueprint-run-owner-gpu-proof` wrapper, trace environment variables, and proof
outputs that must exist before the owner GPU claim can clear. The
scene-asset acquisition files name the missing World Labs/world-manifest and
materialized-asset evidence required before scene/spawn blockers can clear. They
also expose `provider_submission.input_status`; when it is
`ready_for_worldlabs_request_inputs`, the source video is suitable for a World
Labs request before GPU spend. `provider_submission.status` remains
`blocked_missing_worldlabs_api_key` until a shell-only `WORLDLABS_API_KEY` is
configured, and remains `blocked_missing_worldlabs_submission_gate` until
`BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION=true` is also set. The generated
`worldlabs_provider_submission_commands.sh` script checks both values plus source
video preflight before it can call World Labs. Only then does the status become
`ready_to_submit_worldlabs_request`; GPU rental remains blocked until the
generated world and local materialized scene asset exist. The
WebApp handoff files restate the exact upstream-ID, forwarding env, staged
request, and local-rehearsal proof boundary that must clear before the WebApp
return leg is claimable. The GPU VM runtime preflight script checks the VM-side
mount, `nvidia-smi`, owner command executable, Docker availability, and sync
manifest SHA-256 values before the owner command is run; its plan also blocks
when `gpu_vm_sync_manifest.json` is blocked. The GPU VM sync files
inventory required raw, simulation-automation, and run-packet files with size
and SHA-256 checksums so the operator can verify the VM mount or copy before
running the owner command. The packet is a command handoff only; it does not
provision a GPU, call providers, submit WebApp requests, run a simulator,
download assets, copy files, or create owner GPU proof.
The upstream-truth verification script is read-only: it checks accepted
capture, descriptor, and handoff artifacts for real non-placeholder
`site_submission_id`, `request_id`, `buyer_request_id`, and `capture_job_id`,
writes `webapp_upstream_truth_verification_result.json`, and exits blocked
without mutating artifacts when those IDs are missing.
If you do not yet know the owner simulator command, omit `--owner-command`;
the packet will keep `simulator_runtime:missing_simulator_command` blocked
instead of treating the default VM path as real proof.

At this point, the audit should not report `ready_for_owner_gpu_attempt` unless
a WebApp job request has already been staged through Phase 2. The actual owner
GPU proof is still expected to be missing before the first GPU attempt.

## Phase 2: WebApp Forwarding Setup

WebApp site-library requests intentionally use
`/synced-artifacts/sites/<slug>` as their public capture root. Before live
forwarding, map that slug to the Pipeline control-plane capture root:

```bash
export ROBOT_EVAL_JOB_REQUEST_FORWARD_URL="https://<pipeline-host>/api/live-pipeline/job-requests"
export ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN="<redacted>"
export ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED=true
export ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON='{"'"$WEBAPP_SITE_SLUG"'":"'"$CAPTURE_ROOT"'"}'
export BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH="$CAPTURE_ROOT/pipeline/live_pipeline_staged_inputs.json"
```

On the Pipeline host, run the authenticated intake service:

```bash
export BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN="<same-token>"
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765
```

The intake service stages validated WebApp requests. It does not run a simulator
or upgrade proof claims.

After submitting from WebApp, verify the request was staged or queued:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --webapp-job-request /path/to/webapp/robot_eval_job_request.json \
  --stage-webapp-request \
  --staged-inputs-path "$BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH"
```

Treat any WebApp `pipeline_trigger.cpu_pre_gpu_preflight` readiness fields as
advisory request context. The Pipeline-local
`simulation_automation/gpu_handoff_packet.json` is the source of truth for
`ready_for_owner_gpu_preflight`; a WebApp request must not upgrade owner-GPU
readiness, simulator execution, robot policy execution, safety, or public
claims by itself.

Then rerun the first-GPU readiness audit with the staged request pointer:

```bash
blueprint-audit-first-gpu-e2e-readiness \
  --capture-root "$CAPTURE_ROOT" \
  --webapp-site-slug "$WEBAPP_SITE_SLUG" \
  --webapp-staged-inputs "$BLUEPRINT_LIVE_PIPELINE_STAGED_INPUTS_PATH" \
  --simulator isaac_sim \
  --provisioner runpod \
  --simulator-command "$OWNER_SIMULATOR_COMMAND" \
  --simulator-command-location remote
```

For a locally staged rehearsal request, add
`--allow-local-webapp-rehearsal`. Omit that flag for the real WebApp-forwarded
E2E gate so rehearsal evidence cannot pass as live request proof.

When this reports `ready_for_owner_gpu_attempt`, the repo-local inputs, WebApp
handoff configuration, staged WebApp request, simulator gates, and owner command
wiring are all ready for the first GPU attempt. The actual owner GPU proof is
still expected to be missing before that attempt.

For the full E2E test requested through WebApp, also require
`blueprint-audit-first-gpu-cross-repo-readiness` to report
`local_webapp_rehearsal_only_observed=false` and
`full_e2e_webapp_live_forwarding_required_evidence_present=true`. A local
rehearsal can validate request shape, but it is not enough to recommend RunPod
or equivalent GPU spend for the real WebApp-forwarded path.

Stop if any of these blockers appear:

- `webapp:request_capture_root_does_not_match_control_plane`
- `webapp_staged_request:missing_webapp_staged_inputs`
- `webapp_staged_request:webapp_request_not_staged`
- `webapp_staged_request:webapp_request_capture_root_mismatch`
- missing WebApp upstream IDs
- missing or incomplete policy package modality
- rights/privacy scope not cleared for evaluation

## Phase 3: GPU VM Bring-Up

On the GPU VM or pod:

1. Install Docker and NVIDIA Container Toolkit.
2. Confirm `nvidia-smi` works.
3. Pull or build the selected Isaac Sim / Isaac Lab container.
4. Run the Isaac Sim compatibility checker.
5. Mount or sync `$CAPTURE_ROOT` and any materialized scene assets.
6. Warm shader/cache paths before timing the proof command.

The proof wrapper writes:

- `owner_simulator_stdout.log`
- `owner_simulator_stderr.log`

The owner simulator command must write:

- `owner_scene_load_trace.json`
- `owner_spawn_pose_trace.json`
- `owner_action_or_policy_trace.json`
- `owner_artifact_manifest.json`

The command must also exit nonzero on missing scene assets, failed load, failed
spawn, timeout, empty action trace, or missing proof files.

Wrap the owner simulator command with the Pipeline proof runner. The command
receives these environment variables:

- `BLUEPRINT_CAPTURE_ROOT`
- `BLUEPRINT_GPU_PROOF_DIR`
- `BLUEPRINT_SCENE_LOAD_TRACE`
- `BLUEPRINT_SPAWN_TRACE`
- `BLUEPRINT_ACTION_OR_POLICY_TRACE`
- `BLUEPRINT_ARTIFACT_MANIFEST`

The owner command must write those trace/manifest files. The wrapper captures
stdout/stderr, writes `gpu_owner_system_proof.json`, runs the Pipeline validator,
and exits nonzero if proof is incomplete.

```bash
export ISAAC_OWNER_COMMAND="/opt/blueprint/run_isaac_gpu_proof.sh"

blueprint-run-owner-gpu-proof \
  --capture-root "$CAPTURE_ROOT" \
  --proof-dir "$GPU_PROOF_DIR" \
  --owner-system-id "runpod-<pod-id>" \
  --simulator-backend isaac_sim \
  --simulator-version "<isaac-sim-version>" \
  --gpu-model "<gpu-model-from-nvidia-smi>" \
  --operator-id "<operator-id>" \
  --operator-attestation "I ran this command on the owner GPU VM and the referenced traces are from that run." \
  --timeout-seconds 1800 \
  --command "$ISAAC_OWNER_COMMAND"
```

Then rerun simulation automation to ingest the validated proof and refresh the
GPU handoff packet:

```bash
blueprint-run-simulation-automation \
  --capture-root "$CAPTURE_ROOT"
```

That proves only that the owner command ran and returned the required simulator
evidence. Robot readiness remains false.

If you specifically want `simulation_automation/simulator_execution_manifest.json`
to show a command-managed simulator run, set `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`
and pass `blueprint-run-owner-gpu-proof ...` as the `isaac_sim` simulator command.
For the first GPU smoke, the direct wrapper plus proof-ingestion rerun is simpler
and easier to debug.

## Phase 4: WebApp Job Orchestrator GPU Pass

Use this phase when a real WebApp job request and policy package exist. If the
first run is only a scene-load smoke, stay with Phase 3.

If an external operator already allocated a RunPod or other GPU VM, you can label
the request with the selected provisioner. The current non-fixture provisioner
path records a gated request manifest; it does not allocate RunPod by API.

```bash
export BLUEPRINT_ALLOW_GPU_PROVISIONING=true
export BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true
export ISAAC_PROOF_WRAPPER="blueprint-run-owner-gpu-proof --capture-root $CAPTURE_ROOT --proof-dir $GPU_PROOF_DIR --owner-system-id runpod-<pod-id> --simulator-backend isaac_sim --simulator-version <isaac-sim-version> --gpu-model <gpu-model-from-nvidia-smi> --operator-id <operator-id> --operator-attestation owner_gpu_vm_run_attested --timeout-seconds 1800 --command $ISAAC_OWNER_COMMAND"

blueprint-run-robot-eval-job \
  --capture-root "$CAPTURE_ROOT" \
  --job-request-inbox "$CAPTURE_ROOT/pipeline/robot_eval_job_requests/inbox" \
  --provisioner runpod \
  --allow-gpu-provisioning \
  --simulator isaac_sim \
  --allow-simulator-execution \
  --allow-simulator isaac_sim \
  --simulator-command "isaac_sim=$ISAAC_PROOF_WRAPPER" \
  --timeout-seconds 1800 \
  --budget-usd 25
```

Expected job artifacts:

- `pipeline/robot_eval_job_requests/inbox_run_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/gpu_provisioning_request.json`
- `pipeline/robot_eval_jobs/<job_id>/gpu_provisioning_result.json`
- `pipeline/robot_eval_jobs/<job_id>/simulator_result.json`
- `pipeline/robot_eval_jobs/<job_id>/live_eval_closure_manifest.json`

The closure should still block unless live policy execution, real robot POV,
deployment outcomes, signed delivery, safety/contact proof, and review
acceptance are also supplied.

## Phase 5: Proof Audit And Stop Rules

Run:

```bash
blueprint-audit-live-pipeline-setup \
  --capture-root "$CAPTURE_ROOT" \
  --package-dir "$CAPTURE_ROOT/pipeline/robot_eval_jobs/$ROBOT_EVAL_JOB_ID"

blueprint-audit-live-pipeline-proof-boundary \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json

blueprint-audit-live-robot-eval-closure \
  --capture-root "$CAPTURE_ROOT" \
  --job-dir "$CAPTURE_ROOT/pipeline/robot_eval_jobs/$ROBOT_EVAL_JOB_ID"
```

The first GPU milestone is achieved only when the selected simulator result is
`completed`, the owner GPU proof schema validates, stdout/stderr/exit code are
present, and the proof boundary still refuses robot-readiness upgrades.

Stop and fix before continuing if any of these are true:

- capture preflight is blocked
- WebApp capture root does not resolve to the exact Pipeline capture root
- WebApp upstream IDs are missing or placeholder-like
- GPU command exits nonzero
- proof files are missing or not referenced by the artifact manifest
- simulator result says completed but closure audit still reports missing
  simulator evidence
- any public or internal proof boundary upgrades robot readiness from GPU smoke
  alone

## Expected First-Run Failures

The first GPU run should be treated as a discovery pass. The most likely failures
are useful:

- missing or remote-only scene asset references
- Isaac container/driver mismatch
- unsupported GPU for RTX simulation
- scene loads but robot spawn pose is invalid
- action trace is empty or not tied to a scenario run
- WebApp request stages but does not match the active capture root
- policy package exists but no selected modality is executable
- RunPod/provisioner selection is recorded but not automatically allocated

Record the exact failing command, exit code, stdout/stderr paths, missing proof
labels, and closure blockers before changing code.
