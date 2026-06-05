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
  scenario_execution_plan.json
  task_simulation_requests.json
  scenario_simulator_matrix.json
  agent_review_queue.json
  site_eval_director_run_manifest.json
  site_eval_director_proof_boundary.json
  site_eval_director_blocked_manifest.json
  normalized_simulator_attempt_trace.json
  failure_labels.json
  updated_eval_cards.json
  site_eval_prediction_outcome_ledger.json
  site_eval_calibration_report.json
  learned_facility_breakage_library.json
  cosmos_orchestration_exports.json
  site_eval_real_evidence_blocked_manifest.json
  site_eval_fixture_runner_blocked_manifest.json
  agents_sdk_site_eval_director_request.json
  codex_sdk_code_maintainer_request.json
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

The headless robot-eval job orchestrator writes per-job artifacts under:

```text
pipeline/robot_eval_jobs/<job_id>/
  job_request.json
  job_validation.json
  job_plan.json
  agent_orchestration_plan.json
  gpu_provisioning_request.json
  gpu_provisioning_result.json
  simulator_service_request.json
  simulator_service_result.json
  policy_package_manifest.json
  training_request.json
  training_result.json
  evaluation_request.json
  evaluation_result.json
  normalized_attempt_trace.json
  failure_labels.json
  prediction_outcome_ledger.json
  calibration_report.json
  breakage_library.json
  proof_boundary.json
  job_run_manifest.json
  blocked_manifest.json
```

Evaluation prep surfaces these job artifacts as advisory URIs when they already
exist. It does not use them to upgrade simulator, training, robot-readiness, or
public proof fields.

## Site-Eval Director

`blueprint-run-site-eval-director` is a deterministic local director that reads:

- `pipeline/robot_eval_dataset/site_card.json`
- `pipeline/robot_eval_dataset/task_cards.json`
- `pipeline/robot_eval_dataset/scenario_cards.json`
- `pipeline/robot_eval_dataset/eval_cards.json`
- `pipeline/robot_eval_dataset/proof_boundaries.json`
- existing World Labs, Marble, simready, and simulation automation manifests

It writes scenario execution plans, task simulation request manifests, simulator
matrices, fixture-backed normalized attempt traces, success/failure labels,
updated Eval Card views, prediction/outcome ledgers, calibration reports,
learned facility-breakage libraries, Cosmos export/request manifests, review
queues, run manifests, and proof-boundary manifests under
`pipeline/simulation_automation/`.

The fixture runner may execute locally from
`pipeline/robot_eval_inputs/headless_fixture_attempts.json`. It proves only that
the local loop can normalize attempts, label outcomes, calculate calibration
deltas, aggregate breakage records, and write deterministic artifacts. It does
not prove Isaac Sim, MuJoCo, PyBullet, Newton, Cosmos training, real robot
policy execution, safety validation, or public deployment readiness.

If required robot-eval card inputs are missing, the command writes blocked
manifests with `schema_version`, `status=blocked`, `blockers`,
`missing_inputs`, `attempted_commands`, `evidence`, and `claim_boundary`.

Optional `--agents-sdk-site-eval` and `--codex-sdk-code-maintainer` flags write
advisory request manifests only. Missing `openai-agents`, missing Codex SDK,
missing `OPENAI_API_KEY`, or missing `codex mcp-server` are recorded as blocked
advisory manifests. They are not hard failures for deterministic local outputs.
The Codex SDK lane is scoped to implementation diagnosis and code-fix request
manifests only.

## Robot-Eval Job Orchestrator

`blueprint-run-robot-eval-job` is the job-level headless workflow around a
robot-team request. It reads a local request manifest containing customer,
site/package, task/scenario, robot profile, the six robot-team submission
modalities, operation, simulator preference, Cosmos/training preference, budget,
rights/privacy scope, owner system, provenance, and timestamp alignment.

The orchestrator validates rights/privacy and policy evidence, builds or
requires robot-eval dataset cards, requests advisory agent planning, writes GPU
provisioning request/result manifests, runs only allowed fixture or command
simulator paths, writes training request/result manifests, copies Site Eval
Director normalization/calibration/breakage artifacts when fixture evaluation is
used, and writes a final job run manifest plus blocked manifest when any gate or
proof requirement fails.

Agents can choose next deterministic commands, inspect manifests/logs, retry
safe failures, request provisioning, summarize blockers, and route human review.
Agents cannot override rights/privacy blockers, mark simulator/training proof
complete without result/checkpoint manifests, mark robot readiness proven, spend
money, call live providers, or upgrade public claims.

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

Non-fixture GPU provisioning requires both:

- `BLUEPRINT_ALLOW_GPU_PROVISIONING=true`
- `--allow-gpu-provisioning`

Agents SDK robot-eval job orchestration requires:

- `OPENAI_API_KEY`
- `BLUEPRINT_ALLOW_AGENTS_SDK_JOB_ORCHESTRATION=true`
- `--agent-mode agents-sdk`

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

Site-eval director:

```bash
blueprint-run-site-eval-director \
  --capture-root /path/to/capture-root
```

Optional advisory request manifests:

```bash
blueprint-run-site-eval-director \
  --capture-root /path/to/capture-root \
  --agents-sdk-site-eval \
  --codex-sdk-code-maintainer
```

Headless robot-eval job with fixture provisioner and fixture simulator:

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/capture-root \
  --job-request /path/to/robot-eval-job-request.json \
  --job-id fixture-eval-001 \
  --agent-mode fake \
  --provisioner fixture_local \
  --simulator fixture
```

Blocked-by-default command simulator job:

```bash
BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true \
blueprint-run-robot-eval-job \
  --capture-root /path/to/capture-root \
  --job-request /path/to/robot-eval-job-request.json \
  --job-id mujoco-command-001 \
  --provisioner fixture_local \
  --simulator mujoco \
  --allow-simulator-execution \
  --allow-simulator mujoco \
  --simulator-command mujoco='python scripts/run_mujoco_fixture.py'
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
