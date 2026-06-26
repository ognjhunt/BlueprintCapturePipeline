# CPU Pre-GPU Robot-Eval Automation Goal

Date: 2026-06-06

Status: implemented as a scene-agnostic repo-local deterministic contract.

Owner repo: `BlueprintCapturePipeline`

## Objective

After a capture upload completes, Blueprint should go as far as possible without
GPU or paid simulator/provider execution. The CPU/pre-GPU lane must inspect
local scene assets for arbitrary future captures, audit dependencies, estimate a
scene frame, plan real-collider or conservative proxy state, propose
review-required task anchors, compile episode specs, validate multiple spawn
candidates, generate CPU MuJoCo/PyBullet setup artifacts, optionally run
explicitly gated local CPU smoke checks, and emit a GPU handoff packet without
upgrading public proof.

## Deterministic Outputs

The implemented lane writes under `pipeline/simulation_automation/`:

- `scene_asset_inspection.json`
- `scene_asset_inventory.json`
- `scene_asset_dependency_audit.json`
- `scene_asset_preflight.json`
- `scene_frame_estimate.json`
- `collider_proxy_plan.json`
- `cpu_scene_proxy_manifest.json`
- `cpu_preflight_scorecard.json`
- `task_anchor_proposal_manifest.json`
- `episode_spec.v1.json`
- `episode_specs.json`
- `episode_spec_manifest.json`
- `agent_episode_spec_proposals.json`
- `episode_setup_manifest.json`
- `spawn_pose_validation_manifest.json`
- `cpu_preflight_manifest.json`
- `pre_gpu_readiness_summary.json`
- `cpu_simulator_preflight_manifest.json`
- `mujoco_cpu_preflight/*`
- `pybullet_cpu_preflight/*`
- `gpu_handoff_packet.json`
- `gpu_owner_system_proof_schema.json`
- `gpu_run_checklist.md`
- `owner_gpu_simulator_execution_blocked_manifest.json`

`blueprint-run-simulation-automation` and `blueprint-run-robot-eval-job` surface
the CPU preflight and episode setup statuses in their run manifests.

## Proof Rules

Deterministic code owns canonical manifests and proof booleans. Agents may only
propose missing task, anchor, spawn, or review fields with confidence and
provenance. Agent proposals are advisory and cannot mark proof as complete.

Do not claim any of the following without owner-system proof:

- GPU simulator execution
- accepted simulator execution completed
- generated-world rank fidelity or generated-world rank fidelity
- policy success
- physics/contact validation
- off-scope validation
- training completion

Optional local CPU smoke is labeled only as local CPU preflight. Missing
`mujoco` or `pybullet` packages must produce exact install/run instructions, not
a vague failure.

`ready_for_owner_gpu_preflight_handoff` means the CPU side has prepared local
artifacts, dependency warnings, proxy/collider labels, task/spawn proposals, and
owner-system run instructions. It does not mean the scene is ready for robot
evaluation. The only allowed remaining unproven step is actual owner-system GPU
simulator execution, recorded through `gpu_owner_system_proof_schema.json`.

## Commands

```bash
blueprint-build-scene-asset-preflight --capture-root /path/to/capture-root
blueprint-build-episode-specs --capture-root /path/to/capture-root
blueprint-run-cpu-simulator-preflight --capture-root /path/to/capture-root
blueprint-run-simulation-automation --capture-root /path/to/capture-root
```

Owner-system GPU handoff template:

```bash
BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true \
blueprint-run-simulation-automation \
  --capture-root /path/to/capture-root \
  --allow-simulator-execution \
  --allow-simulator isaac_sim \
  --simulator-command isaac_sim='<owner-system Isaac Sim command>'
```

Optional local CPU smoke requires:

```bash
python -m pip install mujoco pybullet

BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true \
blueprint-run-cpu-simulator-preflight \
  --capture-root /path/to/capture-root \
  --allow-cpu-simulator-preflight
```

## Cross-Repo Sync

`BlueprintCapture` handoff contracts may carry task, anchor, robot profile, and
scene asset metadata as advisory preflight inputs while raw capture evidence
remains authoritative.

`Blueprint-WebApp` may display CPU preflight scorecards, dependency warnings,
collider/proxy labels, task/spawn proposals, owner GPU handoff readiness,
episode summaries, preflight blockers, and proof boundaries only as advisory
dataset/setup status. It must not present those fields as robot-ready,
rank-fidelity-scored, safety-validated, simulator-completed, or policy-passed proof.
