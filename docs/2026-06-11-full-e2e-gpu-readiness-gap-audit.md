# Full E2E GPU Readiness Gap Audit - 2026-06-11

## Scope

This audit covers the Capture app, Capture bridge, Pipeline, WebApp robot-eval request path, first-GPU run-packet path, and post-GPU closure boundary before attempting a paid owner-GPU run.

No live provider call, WebApp mutation, GPU provisioning, simulator execution, or rank-fidelity claim was made during this audit.

## Current Decision

Do not rent a RunPod or equivalent GPU VM yet.

The current cross-repo audit output is
`output/2026-06-11-live-product-cross-repo-first-gpu-readiness-current.json`.
Its `gpu_spend_decision.status` is `do_not_rent_gpu_yet`, with
`gpu_rental_recommended_now=false`.

Best current candidate:

`output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611`

That candidate has the local Pipeline handoff and GPU handoff artifacts needed for an owner-GPU preflight handoff, but it is still missing live WebApp/request truth, staged WebApp intake evidence, a launch-order-ready run packet, GPU VM preflight proof, and the explicit owner execution gate.

Update after the isolated owner-GPU fallback smoke: the current accepted owner
proof is `simulator_backend=mujoco` with `robot_asset=procedural_humanoid_proxy`.
That proof is useful generic owner-GPU simulator evidence, but it is not the
requested Isaac Sim robot-asset proof. The regenerated production handoff and
live closure artifacts therefore keep `isaac_sim_execution_proven=false`,
`isaac_robot_asset_execution_proven=false`, and block on
`isaac_sim_unitree_g1_execution_not_proven` when the run packet targets
`isaac_sim`.

Update after the local MuJoCo G1 asset smoke: the Pipeline now also has a local
CPU MuJoCo proof at
`pipeline/simulation_automation/mujoco_g1_local_smoke/mujoco_g1_local_smoke_manifest.json`.
That run loaded the staged World Labs GLB as a converted OBJ, loaded the
official MuJoCo Menagerie Unitree G1 MJCF, used the repo-generated default
`walk_to_target` smoke policy, and captured simulator frames. It proves
`local_mujoco_g1_asset_execution_proven=true`; it still does not prove owner-GPU
execution, Isaac Sim/Lab execution, robot-team locomotion policy quality, real
robot POV, contact/safety readiness, or customer delivery readiness.

The World Labs collider GLB used by this smoke has vertex colors but no embedded
or referenced image textures (`textures_count=0`, `images_count=0`). A
plain/gray/white render in MuJoCo is therefore expected for this converted
collider view and is not equivalent to the original walkthrough video appearance.

Update after the Isaac command packet hardening: the current first-GPU packet now
includes `isaac_unitree_g1_smoke.py` and `run_isaac_unitree_g1_smoke.sh`. The
generated owner command is `bash $PACKET_DIR/run_isaac_unitree_g1_smoke.sh`,
which is intended to run inside Isaac Sim Python on the GPU VM, convert the
staged World Labs GLB to USD, reference the Unitree G1 USD, run the default
`walk_to_target` smoke, capture simulator frames, and write the owner proof
traces. This removes the previous unknown-owner-command gap, but it is still not
executed proof until the VM preflight passes and the command runs on an Isaac
Sim-capable GPU VM.

Update after WebApp forwarding preflight integration: the WebApp repo now writes
a redacted `blueprint.webapp.robot_eval_forwarding_readiness.v1` report with
`npm run pipeline:forwarding:preflight`. Pipeline first-GPU readiness can consume
that report through `--webapp-forwarding-preflight` or
`ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT` as URL/token/capture-root
configuration evidence. This removes the need to copy the forwarding token into
Pipeline audit output, but it still does not prove a submitted WebApp request,
staged WebApp intake, GPU provisioning, simulator execution, or generated-world rank fidelity.

## Green Areas

- Capture-to-Pipeline source contracts: `ready`.
- WebApp-to-Pipeline source contracts: `ready`.
- Pipeline-return source contracts: `ready`.
- Capture bridge cloud tests passed locally with the correct harness.
- WebApp robot-eval request route tests passed locally.
- Pipeline first-GPU/readiness/run-packet focused tests passed locally.
- The latest walkthrough2 capture root includes a local GPU handoff packet with `ready_for_owner_gpu_preflight=true`.

## First-GPU Spend Blockers

These are the seven pre-spend blocker categories emitted by `output/first_gpu_external_input_packet.md`.

### 1. Real WebApp Upstream IDs

Missing non-placeholder IDs:

- `site_submission_id`
- `request_id`
- `buyer_request_id`
- `capture_job_id`

Impact: the capture root is not yet tied to real WebApp/Capture request truth. Do not patch around this with local placeholders; these IDs need to come from the real WebApp/Capture request path.

### 2. WebApp Forwarding Environment

Missing:

- `ROBOT_EVAL_JOB_REQUEST_FORWARD_URL`
- `ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN`
- `ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON`

Expected site mapping for this candidate:

```json
{"first-gpu-walkthrough-2":"output/first-gpu-walkthrough2-storage/local-blueprint/scenes/first-gpu-walkthrough-2/captures/downloads-walkthrough2-20260611"}
```

Impact: the WebApp can validate local request shapes, but this run has not proven live forwarding into the Pipeline intake path.

Current repo-local mitigation: generate
`pipeline/webapp_forwarding_preflight.json` from the WebApp repo and pass it to
`blueprint-audit-first-gpu-e2e-readiness --webapp-forwarding-preflight` or
`blueprint-audit-first-gpu-cross-repo-readiness --webapp-forwarding-preflight`;
also pass the same report to `blueprint-build-first-gpu-run-packet
--webapp-forwarding-preflight` so `first_gpu_webapp_handoff.json` and the
generated handoff verifier carry the same redacted evidence. The report can
clear forwarding-config evidence only when it is ready, redacted, covers the
expected site slug, and has no blockers. It does not clear the staged WebApp
request blocker below.

### 3. Staged WebApp Robot-Eval Request

Missing:

- a real `robot_eval_job_request.v1`
- `pipeline/live_pipeline_staged_inputs.json`

Impact: the Pipeline has no staged real WebApp request envelope for this capture root. The next session should stage one through the guarded intake command, then rerun the cross-repo audit.

The regenerated first-GPU packet now includes
`default_test_robot_eval_job_request.template.json` and
`stage_first_gpu_live_inputs.sh`; those are templates/commands only. They do not
clear this blocker until real WebApp IDs replace the placeholders and the intake
command stages the request.

### 4. WebApp Handoff Packet

Missing ready status:

- `first_gpu_webapp_handoff.json` with `status=ready_for_webapp_handoff_verification`

Impact: the current handoff packet is not ready because the upstream WebApp truth and staged request are not ready.

### 5. First-GPU Run Packet

Current blockers:

- `first_gpu_run_packet_not_ready_for_attempt`
- `launch_order_blocks_gpu_execution`
- blocked launch steps: `webapp_live_handoff`, `gpu_vm_runtime_preflight`, `owner_gpu_simulator_proof`

The regenerated packet exists, but its launch order still blocks GPU execution until WebApp truth, staged request, VM runtime preflight, and explicit gates are settled.

### 6. GPU VM Runtime Preflight

Missing:

- `nvidia-smi` proof on the selected GPU VM
- `gpu_vm_runtime_preflight_result.json` with `status=ready_for_owner_command_attempt`
- a packet whose VM preflight script is safe to run

Impact: no paid VM should be allocated or used until the packet stops blocking the VM preflight plan.

## Owner GPU Simulator Command Binding

No longer a missing pre-spend category in the regenerated packet:

- the run packet contains `owner_default_smoke_command_binding.sh`
- `first_gpu_env.example`, `gpu_vm_commands.sh`, and `gpu_vm_runtime_preflight.sh`
  default `OWNER_SIMULATOR_COMMAND` to
  `bash $OWNER_DEFAULT_SMOKE_COMMAND_BINDING`
- the binding requires real owner runtime commands for scene load, robot spawn,
  and the default walk-to-target policy:
  `OWNER_SCENE_LOAD_COMMAND`, `OWNER_ROBOT_SPAWN_COMMAND`, and
  `OWNER_WALK_TO_TARGET_COMMAND`
- the binding also requires simulator POV evidence:
  `SIM_ROBOT_POV_FRAME_PATH` or `SIM_ROBOT_POV_VIDEO_PATH`
- for `isaac_sim`, the proof runner now defaults the robot asset target to
  Unitree G1 from the Isaac Sim robot assets catalog:
  `Robots/Unitree/G1/g1.usd`
- the owner spawn trace must identify the same Unitree G1 asset; a procedural
  humanoid proxy can be recorded as fallback smoke evidence, but does not clear
  `isaac_sim_execution_proven`

The owner may opt out by setting `BLUEPRINT_USE_DEFAULT_SMOKE_BINDING=false` and
supplying an owner-maintained wrapper, but that wrapper must write the same proof
artifacts.

### 7. Owner GPU Execution Gate

Missing:

- `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`

Impact: this should stay missing until the actual owner-GPU attempt. It is a correct final safety gate, not something to set during audit.

## Candidate-Specific Review Items

- The first-GPU candidate audit found `ready_candidate_count=0` across seven candidates.
- The walkthrough2 candidate is the best current target, but it still blocks on the WebApp/run-packet categories above.
- Spawn validation for walkthrough2 is not a hard local handoff blocker, but it still needs review: 9 valid candidates out of 12, with 3 `spawn_outside_scene_bounds` cases. Pick a valid spawn/task trace before the owner-GPU attempt.
- Capture qualification remains `pre_screen_only` and human review is required. That is acceptable for a pre-GPU handoff, but it is not customer delivery proof.

## Post-GPU Closure Gaps

With the current owner-GPU proof contract, a successful first owner-GPU smoke proves:

- the scene loaded in the selected owner simulator
- the robot spawn pose loaded without immediate invalid state
- the default `walk_to_target` smoke policy wrote a policy execution trace for the
  configured target
- a job-level default `walk_to_target` test policy can now write
  `policy_execution_manifest.json` and `policy_execution_trace.json` with
  `default_test_policy_execution_proven=true` when
  `BLUEPRINT_ALLOW_POLICY_EXECUTION=true` and `--allow-policy-execution` are set
- simulator robot POV evidence was captured as video or frame evidence by the owner
  command; `blueprint-write-owner-gpu-default-smoke-artifacts` can write the policy
  trace and POV manifest after that frame/video exists
- stdout, stderr, exit code, artifact manifest, and operator attestation were preserved
- when the selected backend is `isaac_sim`, the Unitree G1 Isaac USD asset was
  spawned and matched between the proof wrapper and owner spawn trace

It still will not prove:

- Isaac Sim execution when the backend proof is MuJoCo/PyBullet/fixture only
- Unitree G1 asset execution when the spawn trace uses a generated humanoid proxy
- robot-team policy quality beyond the default smoke policy
- owner-run POV evidence; simulator POV is intentionally a separate proof field
- robot-team policy quality or robot-team policy package execution; the generated
  `live_policy_execution_contract.md` defines the separate job-level
  `policy_execution_manifest.json` and `policy_execution_trace.json` evidence
  required for that gate
- real-world validation loop closure
- predicted-vs-actual calibration
- safety/contact/physics readiness
- human review acceptance
- signed customer delivery
- generated-world rank fidelity
- public claim upgrade

After GPU proof exists, run the live robot-eval closure audit and keep real robot POV,
real-world validation, calibration, safety/contact, review acceptance, delivery, and
rank-fidelity as separate gates.

The regenerated first-GPU packet now includes
`real_robot_pov_manifest.template.json`. This makes the required physical POV
shape explicit, but it remains blocked until the manifest contains real robot
camera video refs and action log refs for every required scenario eval run.

## Dirty-State Risk

Current local dirty state spans Pipeline and Capture.

Pipeline dirty files include hardening around agent runtime artifacts, canonical site package behavior, provider preview, qualification, simulation automation, launch bundle tests, first-GPU sample video tests, CPU pre-GPU preflight tests, and a new `tests/test_agent_runtime_artifacts.py`.

Capture dirty files include upstream-ID normalization/rejection work in the upload path, capture bundle support, cloud extract-frame tests, and GPU compatibility docs.

Do not start the paid GPU run from an ambiguous dirty tree. Either land the relevant hardening or explicitly document which dirty changes are part of the run baseline.

## Verification Run During Audit

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Result: 520 passed, 1 skipped.

After adding the fail-closed owner command binding template and default smoke-policy
helper wiring, rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Result: 523 passed, 1 skipped.

After making the generated GPU VM command default to the smoke-policy binding and
regenerating the walkthrough2 run packet, rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 ruff check .
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Result: ruff passed; 525 passed, 1 skipped.

Focused packet/proof regression after the same change:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest tests/test_owner_gpu_proof_runner.py tests/test_simulation_automation.py tests/test_first_gpu_run_packet.py tests/test_cross_repo_first_gpu_readiness.py tests/test_robot_eval_job_orchestrator.py -q
```

Result: 117 passed.

Passed:

```bash
npm test -- --run server/tests/robot-eval-job-requests.test.ts
```

Result: 13 passed.

Passed:

```bash
npm test
```

Run from `BlueprintCapture/cloud/extract-frames`.

Result: 50 passed.

Interrupted:

```bash
xcodebuild test -project BlueprintCapture.xcodeproj -scheme BlueprintCapture -destination 'platform=iOS Simulator,name=iPhone 16e,OS=26.0' -derivedDataPath build/DerivedData -only-testing:BlueprintCaptureTests/CaptureBundleAndInferenceTests
```

This was a redundant focused run while the external alpha gate was also running against the same project/derived-data tree. It was stopped to avoid continued simulator contention and reported `BUILD INTERRUPTED`.

Interrupted after simulator hang:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_external_alpha_launch_gate.py
```

The gate completed the Capture bridge cloud tests and reached the Xcode simulator phase, then slept for more than 17 minutes with no test output. It was stopped and reported `BUILD INTERRUPTED` with `NSMachErrorDomain Code=-308`. Treat this as a blocked local simulator verification item, not as a pass and not as proof of an app-contract failure.

## Next-Session Fix Order

1. Decide the real first-GPU sample: use walkthrough2 unless a better real captured site is available.
2. Produce real WebApp/Capture upstream IDs for that sample; reject placeholders and capture-id-as-upstream fallbacks.
3. Configure WebApp forwarding env for the run, with the capture-root-by-site mapping and token kept out of artifacts.
4. Stage a validated `robot_eval_job_request.v1` into the capture root through Pipeline intake.
5. Regenerate the WebApp handoff packet and first-GPU run packet for the selected simulator/provisioner/owner command.
6. Rerun `blueprint-audit-first-gpu-cross-repo-readiness` until `gpu_rental_recommended_now=true`.
7. Only then allocate the GPU VM, run VM preflight, verify `nvidia-smi`, sync the packet, and set `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true` for the intentional owner command.
8. Save `gpu_owner_system_proof.json`, rerun the cross-repo audit, then run closure/readiness audits before making any rank-fidelity or delivery claim.

## Stop Rules

- Do not allocate paid GPU time while `gpu_rental_recommended_now=false`.
- Do not run `gpu_vm_commands.sh` while its packet says `safe_to_run_now=false`.
- Do not set `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true` during local audit or packet generation.
- Do not claim WebApp live forwarding until a real forwarded request and staged Pipeline intake artifact exist.
- Do not claim scene asset or GPU handoff readiness from generated support artifacts alone; keep raw capture and packet evidence primary.
- Do not claim generated-world rank fidelity from the first GPU smoke.
