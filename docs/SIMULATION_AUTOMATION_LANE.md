# Simulation Automation Lane

Status: fail-closed local orchestration lane.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The simulation automation lane turns persisted capture/package/World Labs/Marble
artifacts into deterministic manifests that plan asset conversion, simulator
execution, training orchestration, evaluation, and proof collection.

It does not call live providers, download remote assets, run simulators, run GPU
training, deploy, send messages, touch payments, or upgrade public claims by
default.

Optional Codex SDK or OpenAI Agents SDK assistance is advisory only. Agents may
propose commands, diagnose failures, summarize traces, and update next-action
plans. Deterministic code owns statuses, gates, manifests, and claim boundaries.

## Inputs

The lane reads existing local artifacts when present:

- `capture_descriptor.json`
- `raw/manifest.json`
- `pipeline/worldlabs_request_manifest.json`
- `pipeline/worldlabs_operation_manifest.json`
- `pipeline/worldlabs_world_manifest.json`
- `pipeline/marble_sim_assets/marble_simready_bridge.json`
- `pipeline/marble_sim_assets/marble_asset_validation.json`
- `pipeline/simready/simready_scene_manifest.json`
- `pipeline/simready/simready_validation.json`
- `pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json`
- `pipeline/cosmos_training_export/manifest.json`

No new World Labs request is made. Marble assets are treated as references until
an explicit local export/conversion manifest or approved simulator run proves
more.

## Outputs

The command writes:

```text
pipeline/simulation_automation/
  simulation_automation_plan.json
  simulation_automation_run_manifest.json
  asset_conversion_plan.json
  simulator_execution_manifest.json
  training_orchestration_manifest.json
  proof_boundary.json
  agent_decision_ledger.json
  simulators/
    isaac_sim_request.json
    isaac_sim_result.json
    mujoco_request.json
    mujoco_result.json
    pybullet_request.json
    pybullet_result.json
    newton_request.json
    newton_result.json
```

Evaluation prep surfaces these artifacts when they already exist. It does not
run this lane automatically and does not turn these artifacts into simulator,
training, or robot-readiness proof.

## Execution Gates

Default behavior is local-only planning:

- `live_provider_calls_performed=false`
- `remote_asset_downloads_performed=false`
- `simulators_run=false`
- `gpu_training_run=false`
- `simulator_execution_proven=false`
- `robot_readiness_proven=false`
- `public_claim_upgrade_allowed=false`

Real simulator execution requires both:

- `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true`
- `--allow-simulator-execution`
- one or more `--allow-simulator <isaac_sim|mujoco|pybullet|newton>`
- explicit `--simulator-command <framework>=<command>`

Cosmos training requires both:

- `BLUEPRINT_ALLOW_COSMOS_TRAINING=true`
- `--allow-training`
- a valid training export and command path, either through
  `--training-command` or the existing Cosmos training runner environment

Missing dependencies or approvals write blocked result manifests. They are not
treated as completion.

## Proof Boundaries

### Local Review Artifacts

Local review artifacts include simready, Marble handoff, robot-eval dataset, and
simulation automation manifests. They can support planning and review only.

They do not prove simulator execution, physics/contact validity, robot policy
success, safety validation, training completion, or public deployment readiness.

### Simulator Execution Proof

Simulator execution proof requires a successful result manifest with command,
stdout, stderr, exit code, artifact paths, and a simulator load/action trace from
the owner simulator.

The lane supports request/result records for Isaac Sim, MuJoCo, PyBullet, and
Newton while keeping each backend replaceable.

### Training Proof

Training proof requires the existing Cosmos LoRA training runner to complete,
write its training run manifest, and produce a checkpoint path. A ready export
manifest or missing-command blocker is not training proof.

### Robot Policy Proof

Robot policy proof requires robot-team-owned assets and action/policy logs from
an accepted simulator or real robot trial. Blueprint capture/package artifacts
alone are not policy proof.

### Safety And Contact Proof

Safety/contact proof requires explicit physics/contact validation logs and
review methodology accepted by the operator or buyer. Collider meshes and local
proxy scenes are only review inputs.

### Public Deployment Readiness

Public deployment readiness requires the complete proof chain above plus
rights/privacy clearance and buyer-approved methodology. This lane must not
upgrade public claims by itself.

## Command

```bash
blueprint-run-simulation-automation \
  --capture-root /path/to/capture-root
```

Optional advisory fake-agent ledger:

```bash
blueprint-run-simulation-automation \
  --capture-root /path/to/capture-root \
  --agent-mode fake
```

Optional blocked-by-default simulator request:

```bash
BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true \
blueprint-run-simulation-automation \
  --capture-root /path/to/capture-root \
  --allow-simulator-execution \
  --allow-simulator mujoco \
  --simulator-command mujoco='python scripts/run_mujoco_fixture.py'
```

Only use real simulator or training commands after confirming the dependency,
approval, cost, GPU, and artifact-storage boundaries for that run.
