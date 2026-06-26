# Newton Physics Backend Implementation Spec

## Decision Summary

Blueprint should implement Newton as a first-class, replaceable GPU physics backend
when the requested proof requires materially better physics than the current local
MuJoCo-first path can provide. Newton should not replace MuJoCo, Isaac Sim, Isaac
Lab-Arena, or PyBullet globally; it should become an explicitly routed backend for
contact-rich manipulation, locomotion at GPU scale, deformables, SDF/hydroelastic
contact, and OpenUSD/Isaac Lab Newton workflows.

The correct relationship is:

- MuJoCo does not "support Newton" as a downstream plugin in Blueprint's current
  architecture.
- Newton integrates MuJoCo Warp / MJWarp as a primary solver/backend in the
  Newton ecosystem.
- Blueprint already models `newton` as a supported simulator id and gated GPU
  physics backend, but does not yet provide Newton-specific runner modules,
  packaged worker images, proof validators, or scenario packets.

## Current Blueprint State

`blueprint_pipeline.robot_eval_job_orchestrator` already includes `newton` in the
canonical simulator list and assigns it a worker image environment variable. Its
provider profile classifies Newton as a GPU physics engine executed through a
gated owner command with an optional `newton` dependency and the standard
`robot_eval_simulator_command_output.v1` contract.

The simulation lane already permits Newton as a real simulator execution target,
but only when all execution gates are explicitly opened:

- `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`
- `--allow-simulator-execution`
- `--allow-simulator newton`
- `--simulator-command newton=<command>`

The lane also records that local review artifacts do not prove simulator
execution, physics/contact validity, policy success, off-scope validation, training
completion, or public generated-world rank fidelity. Newton implementation must preserve
that proof boundary.

## Why Newton May Be Better Physics

Newton is worth implementing when the job needs one or more of these capabilities:

1. **GPU-scale MuJoCo Warp execution**
   - Use when many parallel environments or high-throughput policy evaluation are
     more important than CPU-local smoke-test simplicity.
   - Useful for locomotion and manipulation workloads that already have MJCF or
     MuJoCo-compatible assets but need GPU acceleration.

2. **Contact-rich manipulation**
   - Use when point-contact or simple convex-proxy collision is insufficient.
   - Candidate tasks include insertion, grasp stability, in-hand manipulation,
     close-clearance movement, and contact-sensitive task scoring.

3. **SDF and hydroelastic contact**
   - Use when CAD-like geometry, non-convex collision, or tactile/contact force
     fidelity materially affects the evaluation result.

4. **Closed-chain and high-DOF mechanisms**
   - Use Newton/Kamino-style paths for robotic hands, linkage systems, parallel
     mechanisms, and articulated systems where the default MuJoCo-first proof is
     not expressive enough.

5. **Deformable and granular interactions**
   - Use Newton deformable solvers for cables, hoses, cloth, rubber, packaging,
     granular materials, or terrain/object coupling that rigid-body-only tests
     cannot validate.

6. **OpenUSD / Isaac Lab Newton routing**
   - Use when the scene or robot team requires OpenUSD/Isaac Lab workflows while
     still wanting Newton/MJWarp physics under the same task/reward/reset
     contract.

## Non-Goals

- Do not make Newton the default backend for every robot-eval job.
- Do not remove or weaken MuJoCo-first CPU/low-cost smoke paths.
- Do not mark `rank_fidelity_result_proven`, `physics_contact_validated`,
  `non_ranking_operational_claim_validated`, or `public_claim_upgrade_allowed` true from Newton command
  success alone.
- Do not rewrite raw capture truth, generated scene evidence, or robot-owned
  assets to make Newton outputs appear more authoritative than source captures.
- Do not hardwire Blueprint to NVIDIA-only runtime assumptions in core contracts;
  isolate GPU/Newton assumptions in backend adapters, provider images, and proof
  manifests.

## Routing Policy Additions

Add Newton-specific proof classes to simulator routing and selection policy:

- `gpu_parallel_physics_required`
- `mujoco_warp_required`
- `newton_mjwarp_required`
- `contact_rich_manipulation_required`
- `sdf_collision_required`
- `hydroelastic_contact_required`
- `deformable_object_required`
- `granular_or_mpm_required`
- `closed_chain_mechanism_required`
- `isaac_lab_newton_backend_required`
- `openusd_newton_scene_required`

Selection behavior:

1. Keep MuJoCo as the default first pass when the job is policy/spawn/default-task
   smoke and does not require Newton-only capabilities.
2. Recommend Newton when any Newton-specific proof class is present and `newton`
   is in the request's allowed backend list.
3. Recommend Isaac Sim or Isaac Lab-Arena instead when the required proof is
   Isaac-specific, RTX-sensor-specific, Arena-batch-specific, or owner-scored via
   Isaac/Arena artifacts.
4. Emit a non-blocking warning when Newton is requested but the job lacks a GPU
   provisioner, Newton worker image, or accepted owner command.

## Required New Modules and Entrypoints

Add these modules behind optional dependencies and execution gates:

1. `blueprint_pipeline.newton_worker_runtime_preflight`
   - CLI: `blueprint-run-newton-worker-runtime-preflight`
   - Responsibilities:
     - import Newton/Warp/MuJoCo Warp when available;
     - record versions, CUDA availability, GPU name, driver/runtime metadata;
     - run a tiny deterministic rigid-body step if dependencies are installed;
     - write a blocked manifest when dependencies, GPU, or approvals are absent.

2. `blueprint_pipeline.newton_simulator_command`
   - CLI: `blueprint-run-newton-simulator-command`
   - Responsibilities:
     - execute a Newton native, Isaac Lab Newton, or MJWarp scene command;
     - capture stdout/stderr/exit code;
     - normalize output into `robot_eval_simulator_command_output.v1`;
     - require explicit `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true` and
       `--allow-simulator-execution` when invoked through orchestration.

3. `blueprint_pipeline.newton_contact_rich_manipulation_command`
   - CLI: `blueprint-run-newton-contact-rich-manipulation`
   - Responsibilities:
     - build or load a contact-rich task packet;
     - run an SDF/hydroelastic or MJWarp contact scenario when available;
     - export contact traces, pass/fail metrics, and visual review artifacts;
     - keep readiness proof false unless an accepted owner proof bundle upgrades
       specific simulator-execution evidence.

4. `blueprint_pipeline.newton_proof_validation`
   - Responsibilities:
     - validate Newton proof packages;
     - reject unsupported readiness/safety/public-claim upgrades;
     - verify solver metadata, contact mode, scene provenance, action trace,
       artifact manifest, and owner attestation.

## Artifact Contracts

### Runtime Preflight Manifest

Path recommendation:

`pipeline/simulation_automation/newton_runtime_preflight/newton_runtime_preflight_manifest.json`

Required fields:

- `schema_version`: `newton_runtime_preflight.v1`
- `status`: `passed | blocked | failed`
- `newton_import_available`
- `warp_import_available`
- `mujoco_warp_import_available`
- `cuda_available`
- `gpu_name`
- `driver_version`
- `runtime_versions`
- `smoke_steps_requested`
- `smoke_steps_completed`
- `blockers`
- `claim_boundary`

### Newton Simulator Output

Path recommendation:

`pipeline/simulation_automation/newton_command/newton_simulator_output.json`

Required fields:

- `schema_version`: `robot_eval_simulator_command_output.v1`
- `simulator_backend`: `newton`
- `newton_execution_mode`: `native_newton | isaac_lab_newton | isaac_sim_newton | mujoco_warp`
- `solver`: `mujoco_warp | kamino | xpbd | featherstone | semi_implicit | vbd | mpm | mixed`
- `scene_format`: `mjcf | urdf | openusd | generated_fixture | unknown`
- `contact_model`: `point_contact | sdf | hydroelastic | deformable_coupled | unknown`
- `command`
- `stdout_uri_or_path`
- `stderr_uri_or_path`
- `exit_code`
- `scene_load_trace_uri_or_path`
- `spawn_trace_uri_or_path`
- `action_trace_uri_or_path`
- `contact_trace_uri_or_path`
- `artifact_manifest_uri_or_path`
- `pass_fail_criteria`
- `accepted`
- `blockers`
- `claim_boundary`

### Claim Boundary Defaults

Newton artifacts must default to:

```json
{
  "simulator_execution_proven": false,
  "newton_execution_proven": false,
  "physics_contact_validated": false,
  "robot_policy_execution_proven": false,
  "rank_fidelity_result_proven": false,
  "non_ranking_operational_claim_validated": false,
  "public_claim_upgrade_allowed": false,
  "real_robot_contact_proven": false
}
```

Only deterministic accepted artifacts may upgrade simulator-specific booleans,
and even accepted Newton execution must not automatically upgrade generated-world rank fidelity,
safety, public claims, or generated-world rank fidelity.

## Provider and Packaging Requirements

Add Newton-specific provider support without changing existing MuJoCo behavior:

- `deploy/docker/robot_eval_worker/newton/Dockerfile`
- environment variable: `BLUEPRINT_NEWTON_EVAL_WORKER_IMAGE_REF`
- optional cache env vars:
  - `BLUEPRINT_NEWTON_ASSET_CACHE`
  - `BLUEPRINT_NEWTON_PACKAGE_CACHE`
  - `BLUEPRINT_MUJOCOWARP_ASSET_CACHE`
- provider input setup should emit Newton build/push/run instructions when
  `--simulator newton` is selected.
- GPU provider launcher should preserve existing proof boundaries and cost-control
  ledger behavior.

## Test Plan

Minimum test coverage:

1. Selection policy recommends Newton when Newton-only proof classes are present.
2. Selection policy keeps MuJoCo first for generic smoke jobs.
3. Newton simulator command parsing requires explicit command and allow-listing.
4. Missing Newton dependencies write blocked manifests, not false success.
5. Accepted Newton command output can prove `newton_execution_proven` only inside
   simulator-execution proof boundaries.
6. Newton proof validation rejects attempts to set generated-world rank fidelity, safety, public
   claim upgrade, or real-robot contact proof directly.
7. Provider input setup emits `BLUEPRINT_NEWTON_EVAL_WORKER_IMAGE_REF` and Newton
   Docker instructions for `--simulator newton`.
8. Documentation examples show native Newton, Isaac Lab Newton, and MJWarp modes
   as separate execution modes.

## Implementation Phases

### Phase 0: Documentation and policy spec

- Land this spec.
- Add docs explaining the MuJoCo/Newton relationship: Newton integrates MuJoCo
  Warp; MuJoCo-first remains the default low-cost path.

### Phase 1: Routing and blocked manifests

- Add Newton proof classes to routing policy.
- Add tests proving Newton recommendation only when capability classes require it.
- Add blocked preflight manifest support without requiring Newton to be installed.

### Phase 2: Runtime preflight

- Implement `newton_worker_runtime_preflight`.
- Record import/GPU/version metadata.
- Keep all readiness booleans false.

### Phase 3: Generic Newton command runner

- Implement `newton_simulator_command`.
- Normalize native Newton, Isaac Lab Newton, Isaac Sim Newton, and MJWarp outputs
  into the standard simulator command contract.

### Phase 4: Contact-rich manipulation proof

- Implement a minimal Newton contact-rich scenario packet.
- Add contact trace and pass/fail metrics.
- Compare against MuJoCo baseline when the same MJCF asset exists.

### Phase 5: Deformable and hydroelastic extensions

- Add SDF/hydroelastic contact task support.
- Add deformable cable/hose/cloth scenarios only after rigid contact-rich proof is
  stable.

### Phase 6: Provider image and owner GPU execution

- Add Newton Dockerfile/provider image support.
- Add runpod/vast/gcp examples where applicable.
- Validate owner proof ingestion without weakening readiness boundaries.

## Acceptance Criteria

Newton implementation is acceptable when:

- generic jobs still default to MuJoCo-first low-cost routing;
- Newton is recommended for explicit Newton-capability proof classes;
- Newton can produce blocked, failed, and accepted manifests deterministically;
- Newton artifacts preserve raw capture truth and do not overclaim readiness;
- Newton runner supports at least one real execution mode on an owner GPU;
- tests cover routing, proof boundaries, blocked dependencies, and provider setup;
- docs clearly explain when Newton is better physics and when it is unnecessary.
