# Fable execution instruction

Start from `origin/main` at `d1220f788acb3b5d263af36297ed58443772469a`. Do not
reimplement the merged startup modules. First wire the supervisor, review canary, asset
gate, phase trace, and teardown transaction into the real default launcher; complete
P0-3 build/live validation; reconcile the duplicate stance configurators; then close
kitchen gaps A-F. The arm visual-center, RGB integrity, and strict scorer repairs
described as dirty-local are absent from main and must be carefully ported rather than
overwriting newer main changes. Finish with green CI and live provider proof.

For live validation, Fable will additionally need the selected scenario/task artifacts,
attempt 023/025 logs and media, the content-addressed kitchen archive, image registry
access, and file-mounted provider/object-store secrets. Code-only work does not require
your local machine.

# Handoff 2: Random kitchen-task live E2E

Date: 2026-07-10

Repository baseline: `origin/main` at
`d1220f788acb3b5d263af36297ed58443772469a`

Fresh run root: `output/kitchen_random_task_e2e_20260710T131557Z/`

## Objective and truth boundary

Produce a fresh live-provider simulation episode proving together:

- The complete kitchen assets existed inside the worker.
- A Unitree G1 was correctly spawned, supported, collision-free, target-facing,
  reachable and visible.
- The configured Unitree locomotion controller owned navigation.
- Fresh GR00T N1.7 `UNITREE_G1_SONIC` actions owned manipulation.
- OSCAR/WAM remained an action-conditioned evaluator/support model.
- Every transition used the current observation and a fresh policy action.
- The exact action drove the controller/FK/skeleton route and carried state.
- The episode stopped because the registered task criterion passed.
- Full episode media passed strict full-horizon review.
- Provider result, freshness, spending and teardown were proven.

This is sim-only. Never promote it into physical G1 readiness or real-world task
success.

## Selected task

The selected task remains:

- Task: `microwave_door`
- Target: `/root/Microwave017`
- Affordance: `/root/Microwave017/Microwave017_Door`
- Reference stance: `[-1.229635, 1.471274, 0.84]`
- Reference yaw: `3.141593`
- Registered completion: microwave-door articulation angle increase of at least
  `0.35 rad`

Do not reroll this task merely because it is difficult.

## Historical live evidence

### Attempt 023

DigitalOcean RTX 6000 Ada.

Proved:

- Complete 185-file kitchen bundle.
- Isaac Sim 6 startup.
- Renderable Unitree G1 geometry.
- Task-specific stance execution.
- RTX media production.

Failed:

- Manipulation geometry gate.
- Review-camera quality gate.

### Attempt 025

DigitalOcean RTX 6000 Ada.

The worker completed after startup, but:

- Arm reach used link-transform origins rather than rendered visual meshes.
- Reported shoulder and palm heights were physically wrong.
- Dedicated verification render was completely black.
- Old overview and top-down views were not acceptable review evidence.

Teardown returned provider `not_found`; continuing inventory was zero.

### Later capacity attempts

- DigitalOcean RTX capacity disappeared.
- RunPod RTX A6000/A40 creates returned capacity-related HTTP 500 responses before a
  pod ID.
- Reduced-disk RunPod retry failed the same way.
- These create failures incurred no GPU spend.
- Latest preserved final inventory showed zero resources and $0/hour continuing burn.

Capacity is time-sensitive. Re-run read-only DO and RunPod probes before any paid retry.

## Dirty-local repairs absent from main

The following behavior exists in the local dirty worktree but is not fully present on
`origin/main`.

Do not replace newer main files wholesale. Port these changes with regression tests.

### Runner geometry and media repairs

File:

`scripts/run_isaac_g1_kitchen_parity_eval.py`

Required behavior:

- Select canonical palm, distal wrist and shoulder links.
- Do not accidentally select finger links as the palm.
- Traverse USD instance proxies.
- Derive shoulder/wrist/palm geometry from rendered visual-mesh centers.
- Use visual-mesh centers for manipulation arm posing and reach measurement.
- Reject black, white, flat, empty, malformed and non-finite RGB frames.
- Write frame-integrity sidecars.
- Produce explicit corrupt-frame blockers.
- Use a task-framed third-person overview as fallback when the dedicated verification
  render is corrupt.
- Never use the schematic layout as review evidence.

### Provider spend/startup repairs

File:

`src/blueprint_pipeline/isaac_g1_kitchen_parity_job.py`

Required behavior:

- Calculate spend admission from the selected provider/GPU rate.
- Do not use an unrelated marketplace maximum rate.
- Support current DO and RunPod capacity schemas.
- Permit large split-layer images when the largest layer and timeout are bounded.
- Continue rejecting giant unsplit image layers.
- Count every sequential retry in the cumulative spending reservation.

### Strict evaluator contract repairs

Files:

- `src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py`
- `src/blueprint_pipeline/oscar_cosmos_wam_evaluator.py`

Required behavior:

- Carry exact normalized action vector, checksum, dimension and timing.
- Carry controller/FK state and generated state.
- Require recovered action and checksum.
- Require numeric per-dimension error and uncertainty.
- Require calibration identity.
- Require threshold and units.
- Require evidence references and the termination chunk.
- Reject boolean-only scorer responses.
- Reject missing, wrong-dimensional, untimed, non-finite or unit-invalid actions.

Main's `episode_trace_consistency.v1` does not satisfy this requirement. It is a
graded root-trace support score, not strict action-aware forward/inverse recovery.

## Blocker A: provider arm geometry must pass live

Attempt 025 observed approximately:

- Link-origin shoulder height: `0.175 m`
- Rendered shoulder visual center: `1.129 m`
- Link-origin hand height: `0.005 m`
- Rendered palm visual center: `0.924 m`

The old gate measured the wrong geometry.

Required implementation:

- Traverse instance proxies below arm link prims.
- Compute world-space bounds of rendered mesh descendants.
- Use those centers for shoulder, elbow, wrist and palm landmarks.
- Fail closed if visual meshes exist but only link origins are available.
- Record the exact prim path and measurement source for every landmark.

Acceptance:

- Physically plausible arm heights.
- Stance, collision, clearance, facing and reach gates all pass.
- Robot POV visibly contains the microwave affordance and active arm.
- Two distinct reach actions result in distinct FK/visual conditioning.
- Fresh live Isaac validation succeeds.

## Blocker B: third-person and full-horizon media

Required implementation:

- Fresh third-person Isaac RGB shows G1, microwave, floor support, orientation and
  clearance.
- Every output frame passes black/white/flat/non-finite integrity checks.
- Verify dimensions, exposure, clipping and robot/target occupancy.
- Verify every frame of the ordered episode—not only the first or final clip.
- Detect stale, duplicated or reordered frames.
- Create contact sheets only from the full ordered episode.
- Semantic review must support abstention.
- Missing or invalid semantic-review API is blocked, not passed.
- Never discard a garbage tail to manufacture success.

## Blocker C: official controller/FK bridge is missing

The sealed launch requires `--action-skeleton-command`, but the launch plan does not
supply it.

The existing sim2sim command explicitly is not the official WBC controller route.

Required implementation:

- Use the pinned official GR00T WholeBodyControl/SONIC stack.
- Consume the exact normalized and timed `UNITREE_G1_SONIC` action chunk.
- Verify the action SHA-256.
- Apply it through the actual controller and robot model/FK.
- Return the required `sc3_controller_fk_runtime_result.v1` result.
- Return fresh runtime ID, controller/model hashes, numerical landmarks and
  proprioception.
- Carry the returned state into the next observation.
- Prove two action perturbations result in different FK skeletons/state.

Do not use shape-only projection, fixture action, zero action, replay or the
non-official smoke command.

## Blocker D: persistent Isaac transition and completion evaluator

The sealed plan requests stop-on-completion but does not pass:

- `--task-success-contract`
- `--task-completion-command`

More importantly, the current WAM loop does not expose a persistent microwave
articulation state changed by the G1 action.

Required implementation:

- Materialize the microwave success contract into the sealed input bundle.
- Maintain one persistent Isaac episode across all policy/WAM steps.
- Apply every controller result to the same simulator state.
- Record microwave door angle before and after every action.
- Bind every measurement to timestamp, action SHA and artifact hash.
- Implement the typed signed completion result required by the evaluator.
- Stop only after the door angle increases by at least `0.35 rad`.
- Enforce minimum applicable transitions.
- A watchdog, step cap, process exit, clip end, frame count or root proximity is
  blocked/timeout—not success.

## Blocker E: sealed-plan fixture contradiction

`build_sealed_launch_plan` still contains:

`--harness-backend-kind fixture`

It also does not supply the controller/FK or completion commands.

Required implementation:

- Use the real perception/runtime backend in the strict live lane.
- Pass the real action-skeleton command.
- Pass the task-success contract.
- Pass the task-completion command.
- Require fresh learned-policy requery every applicable step.
- Query GR00T before the first manipulation transition.
- Do not emit an initial deterministic manipulation fallback before the first fresh
  policy result.
- Keep locomotion and manipulation ownership explicit.
- Keep OSCAR/WAM labeled solely as evaluator/support.

Acceptance:

- One fresh policy runtime-result ID per applicable transition.
- One current observation checksum per transition.
- Exact action SHA matches policy, controller/FK, WAM conditioning, carried state,
  consistency request and task measurement.
- No fixture, replay, deterministic manipulation fallback, synthetic action or stale
  action appears in the accepted manifest.

## Blocker F: real strict action-aware scorer

Required implementation:

- Recover the timed action from generated motion.
- Return numeric error and uncertainty per action dimension.
- Identify calibration and units.
- Validate the termination chunk.
- Validate action checksum and timing.
- Run forward and inverse checks independently.
- Fail on missing fields, wrong dimensions, non-finite data, unit mismatch, fabricated
  checksum or threshold exceedance.

Semantic video review must remain a separate judge.

The local visual-motion smoke may provide diagnostics only. It cannot prove
forward/inverse consistency.

## Blocker G: startup reliability integration

Before the next paid kitchen attempt:

1. Use the atomic startup supervisor.
2. Run the fast startup canary.
3. Run the review-renderer canary.
4. Reuse the same passing warm allocation.
5. Run the kitchen asset gate before Isaac.
6. Run the adaptive stance loop in the real scene.
7. Start policy/controller/simulator services.
8. Execute the full episode.
9. Upload artifacts.
10. Delete the allocation and verify zero residual inventory.

Do not manually stitch together independent startup commands and leave teardown
ownership ambiguous.

## What Fable needs in cloud

### Code-only work

- Repository at the exact main SHA.
- Both handoffs.
- Python 3.12 and development dependencies.
- The local dirty kitchen behavior described above, supplied as a carefully ported
  patch or source reference.
- Do not paste or commit secrets.

### Geometry and media diagnosis

Provide:

- `random_task_selection_reroll_002.json`
- `selected_isaac_scenario_attempt_014.json`
- Attempt 023 render output, runner log and manifest.
- Attempt 025 render output, runner log, manifest and frame-integrity evidence.
- The selected task/stance manifests.
- Any offline visual-mesh geometry diagnostic.

### Kitchen/provider execution

Provide:

- Complete `Collected_KitchenRoom` tree or a fresh signed archive.
- Expected inventory and checksums.
- Registry pull access to both digest-pinned images.
- File-mounted DO/RunPod and object-store credentials.
- Explicit paid-execution authorization, spending cap and one-resource limit.

Expected kitchen reference:

- 185 files.
- Approximately 1.24 GB materialized.
- Approximately 699 MB archived.
- Main USD: `Collected_KitchenRoom/KitchenRoom.usd`

### Official controller work

Provide:

- Pinned GR00T WholeBodyControl source and submodules.
- SONIC controller/deploy assets and model files.
- Exact WBC revision:
  `6d8e931b9b10a4db2d8e7aba3ad6d5da3529ff3b`
- Exact Isaac-GR00T revision:
  `e5749287857afd97b78f1147166137de29746392`
- GR00T N1.7 SONIC checkpoint access.
- Access to the sealed image or its non-secret `/opt/wbc` and `/opt/gr00t`
  contents.

The current sealed image merely clones WBC source. It does not prove that the official
simulator/controller environments, Git LFS assets and C++ controller are installed and
runnable.

## Final pass criteria

Do not report completion until all are true:

- The selected microwave task was not silently replaced.
- Full kitchen bundle was provider-verified.
- G1 stance and camera evidence passed live.
- Fresh GR00T SONIC actions controlled manipulation.
- Real controller/FK results conditioned every transition.
- State was carried forward.
- The persistent microwave door angle increased by at least `0.35 rad`.
- Episode termination came from that registered criterion.
- Full ordered robot-POV and third-person media passed.
- Strict forward and inverse consistency passed independently.
- Full-episode semantic review passed independently.
- All artifacts are fresh, checksummed and nonce-bound.
- Spending is within the authorized goal cap.
- Provider resources and storage are deleted.
- Final inventory proves zero continuing spend.
- Focused tests and the broad relevant CI lane pass.
- Reporting remains explicitly sim-only.
