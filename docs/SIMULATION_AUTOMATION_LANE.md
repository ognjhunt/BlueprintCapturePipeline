# Simulation Automation Lane

Status: fail-closed local orchestration lane.

Owner repo: `BlueprintCapturePipeline`

## Purpose

The simulation automation lane turns persisted capture/package/World Labs and
CPU-preflight artifacts into deterministic manifests that plan asset conversion,
simulator execution, training orchestration, evaluation, and proof collection.

It does not call live providers, download remote assets, run simulators, run GPU
training, deploy, send messages, touch payments, or upgrade public claims by
default.

Optional Codex SDK or OpenAI Agents SDK assistance is a gated live-operator
surface, not a request-manifest-only surface. When the SDK, credential, CLI, and
environment gates are open, Agents SDK operators may inspect manifests/logs,
choose next deterministic commands, trigger allowed reruns, route review,
summarize blockers, and maintain progress ledgers. Codex SDK operators may
diagnose failures, patch code, run tests, and produce diffs when pipeline
failures require code changes. Deterministic code owns statuses, validation,
packaging, checksums, rerun policy, manifests, and claim boundaries.

## Inputs

The lane reads existing local artifacts when present:

- `capture_descriptor.json`
- `raw/manifest.json`
- optional local `PLY`, `USD`, `USDA`, or `USDC` scene assets supplied with `--scene-asset`
- `pipeline/worldlabs_request_manifest.json`
- `pipeline/worldlabs_operation_manifest.json`
- `pipeline/worldlabs_world_manifest.json`
- `pipeline/simulation_automation/scene_asset_inspection.json`
- `pipeline/simulation_automation/scene_frame_estimate.json`
- `pipeline/simulation_automation/cpu_preflight_scorecard.json`
- `pipeline/simulation_automation/episode_spec_manifest.json`
- `pipeline/simulation_automation/cpu_simulator_preflight_manifest.json`
- `pipeline/simulation_automation/arena_environment_packet.json`
- optional `pipeline/simulation_automation/gpu_owner_system_proof.json`

Legacy/advisory artifacts are read when present, but are not part of the active
default process:

- `pipeline/marble_sim_assets/marble_simready_bridge.json`
- `pipeline/marble_sim_assets/marble_asset_validation.json`
- `pipeline/simready/simready_scene_manifest.json`
- `pipeline/simready/simready_validation.json`
- `pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json`
- `pipeline/cosmos_training_export/manifest.json`

No new World Labs request is made. Already-generated Marble CDN assets may be
materialized before this lane through `blueprint-materialize-worldlabs-assets`,
which writes local checksum/provenance files and `pipeline/worldlabs_export_manifest.json`.
Those local GLB/PLY/USD files are still review inputs until an approved
simulator run proves more.

## Outputs

The command writes:

```text
pipeline/simulation_automation/
  simulation_automation_plan.json
  simulation_automation_run_manifest.json
  scene_asset_inventory.json
  scene_asset_dependency_audit.json
  scene_asset_preflight.json
  scene_asset_inspection.json
  scene_frame_estimate.json
  collider_proxy_plan.json
  cpu_scene_proxy_manifest.json
  cpu_preflight_scorecard.json
  task_anchor_proposal_manifest.json
  episode_spec.v1.json
  episode_specs.json
  episode_spec_manifest.json
  agent_episode_spec_proposals.json
  episode_setup_manifest.json
  spawn_pose_validation_manifest.json
  cpu_preflight_manifest.json
  pre_gpu_readiness_summary.json
  cpu_simulator_preflight_manifest.json
  cpu_simulator_preflight_README.txt
  arena_environment_packet.json
  gpu_handoff_packet.json
  gpu_owner_system_proof_schema.json
  owner_gpu_simulator_execution_proof_manifest.json
  gpu_run_checklist.md
  owner_gpu_simulator_execution_blocked_manifest.json
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
    isaac_lab_arena_request.json
    isaac_lab_arena_result.json
    mujoco_request.json
    mujoco_result.json
    pybullet_request.json
    pybullet_result.json
    newton_request.json
    newton_result.json
  mujoco_cpu_preflight/
    episode_scene.xml
    smoke_result.json
    blocked_manifest.json
  pybullet_cpu_preflight/
    episode_scene.urdf
    smoke_result.json
    blocked_manifest.json
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
  policy_adapter_manifest.json
  training_request.json
  training_result.json
  evaluation_request.json
  evaluation_result.json
  arena_eval_schedule.json
  arena_eval_retry_queue.json
  arena_eval_cost_ledger.json
  arena_eval_resume_manifest.json
  arena_result_ingest_ledger.json
  arena_artifact_checksums.json
  arena_eval_metrics.json
  normalized_attempt_trace.json
  failure_labels.json
  clips_manifest.json
  rollout_vision_labels.json
  review_resolution_ledger.json
  accepted_failure_labels.json
  prediction_outcome_ledger.json
  calibration_report.json
  breakage_library.json
  arena_rerun_plan.json
  arena_rerun_lineage.json
  customer_handoff_report.md
  customer_handoff_report.json
  delivery_manifest.json
  signed_access_manifest.json
  live_operator_ledger.json
  dataset_card.json
  license_manifest.json
  package_index.json
  checksums.json
  archive_manifest.json
  post_training_data_package_export_manifest.json
  proof_boundary.json
  job_run_manifest.json
  blocked_manifest.json
```

Evaluation prep surfaces these job artifacts as advisory URIs when they already
exist. It does not use them to upgrade simulator, training, robot-readiness, or
public proof fields.

When `blueprint-run-robot-eval-job` is run with `--simulator isaac_lab_arena`
or `--arena-results-dir`, the job also runs the Arena result ingest/package
lane. Existing rollout artifacts can feed `evaluation_result.json` through
`normalized_attempt_trace.json` and `failure_labels.json`, while
`simulator_execution_proven`, `robot_policy_execution_proven`,
`robot_readiness_proven`, safety/contact proof, and public-claim upgrades remain
false unless accepted owner evidence separately proves those claims.

## CPU Pre-GPU Lane

The CPU/pre-GPU lane runs before any GPU/provider execution. It writes
deterministic manifests that are safe to sync to WebApp as advisory status:

- `scene_asset_inventory.json` records local PLY, USD/USDA/USDC, GLB/GLTF, OBJ,
  URDF, and MJCF/XML assets with size and checksum.
- `scene_asset_dependency_audit.json` records USD sublayers, references,
  payloads, textures/material paths, GLTF buffers/images, OBJ material libs, and
  URDF/MJCF mesh refs. Missing local files and remote refs are warnings or
  blockers; the lane never downloads them.
- `scene_asset_inspection.json` parses local PLY headers/bounds, USD metadata,
  GLTF/GLB metadata, OBJ vertex bounds, and URDF/MJCF collision metadata.
- `scene_frame_estimate.json` estimates bounds, centroid, floor, and up-axis when
  enough local evidence exists.
- `collider_proxy_plan.json` labels `real_collider_proven`,
  `proxy_estimated`, `missing_collider`, and `review_required`.
- `cpu_scene_proxy_manifest.json` records conservative floor/bounds/object proxy
  geometry for CPU spawn sanity checks only.
- `cpu_preflight_scorecard.json` splits backend proof labels such as
  `isaac_usd_import_candidate`, `isaac_usd_collision_unverified`,
  `portable_collider_glb_missing`, `cpu_proxy_collision_estimated`, and
  `simulator_execution_not_run`.
- `task_anchor_proposal_manifest.json` proposes review-required task anchors from
  task cards, capture metadata, task hypotheses, scene class hints, object
  labels, and scene asset semantic names.
- `episode_spec.v1.json` compiles scene/task/scenario/robot-profile inputs into
  review-required episode setup specs.
- `episode_specs.json` is a compatibility alias for the compiled episode specs.
- `agent_episode_spec_proposals.json` is a review-input proposal surface. Agents
  may propose missing anchors, spawn fields, or task fields with
  confidence/provenance, but cannot set proof booleans.
- `episode_setup_manifest.json` and `cpu_simulator_preflight_manifest.json`
  describe generated CPU MuJoCo/PyBullet fixtures and optional smoke status.
- `spawn_pose_validation_manifest.json` checks multiple spawn candidates for
  finite coordinates, floor consistency, scene bounds, suspicious scale, and
  overlap with known/proxy geometry where metadata exists.
- `arena_environment_packet.json` translates Site, Task, Scenario, and Eval
  Cards plus `episode_spec.v1.json` into a proof-bounded Isaac Lab-Arena review
  package with Scene, Embodiment, Task, scenario, metric/eval, and episode
  bindings. It is a package/spec artifact only; it does not prove Isaac
  Lab-Arena imported the scene, ran a policy, validated contact, or proved robot
  readiness.
- `cpu_preflight_manifest.json` and `pre_gpu_readiness_summary.json` summarize
  `ready_for_owner_gpu_preflight_handoff`. That phrase means local CPU checks
  and deterministic handoff artifacts are ready for an owner GPU run. It does
  not mean robot evaluation is ready.
- `gpu_handoff_packet.json`, `gpu_owner_system_proof_schema.json`, and
  `gpu_run_checklist.md` tell the owner system what backend to run, what env vars
  and commands to use, what logs to capture, and what pass/fail criteria apply.
- If `gpu_owner_system_proof.json` is supplied, the lane validates simulator
  logs, scene load trace, spawn trace, action/policy trace, artifact manifest,
  pass/fail criteria, and owner attestation before writing
  `owner_gpu_simulator_execution_proof_manifest.json`.
- `owner_gpu_simulator_execution_blocked_manifest.json` is intentionally present
  until owner-system GPU simulator proof is supplied.

Generated default robot profiles are review fixtures only:

- `mobile_manipulator_rgbd_fixture`
- `differential_drive_rgbd_fixture`
- `humanoid_rgbd_fixture`

Missing optional CPU simulator packages do not block deterministic manifest
generation. They write exact blocker/install/run instructions:

```bash
python -m pip install mujoco pybullet

BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true \
blueprint-run-cpu-simulator-preflight \
  --capture-root /path/to/capture-root \
  --allow-cpu-simulator-preflight
```

The optional smoke runner uses only local CPU paths: PyBullet `DIRECT` and
MuJoCo compile/step. A passing local CPU smoke may be displayed only as
`local CPU preflight smoke`; it is not owner-system simulator execution, robot
readiness, policy success, physics/contact validation, or safety proof.

The GPU handoff packet generally recommends Isaac Sim first when rich USD or
OpenUSD-like scene assets exist. `isaac_lab_arena` appears as an optional
composable policy-eval package lane alongside Isaac Sim when an Arena packet can
bind the Blueprint scene, embodiment, task, scenario, and eval components.
MuJoCo and PyBullet are recommended only for compatible MJCF/URDF assets or
generated/proxy fixtures. The owner system must provide simulator stdout,
stderr, exit code, scene load trace, spawn trace, artifact manifest, action or
policy trace when applicable, and an owner attestation before any simulator
execution proof can be upgraded.

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
SDK operator manifests. With `OPENAI_API_KEY`, the matching SDK dependency,
`--allow-live-agents-sdk-operator` or `--allow-live-codex-sdk-operator`, and the
matching environment gate, they execute as live operators and log decisions,
tool-call summaries, chosen commands, refusals, blockers, and proof effect.
Missing SDKs, credentials, CLI gates, or environment gates are recorded as
blocked operator manifests. They are not hard failures for deterministic local
outputs. The Codex SDK lane is scoped to implementation diagnosis, code patches,
focused tests, and diff summaries.

## Robot-Eval Job Orchestrator

`blueprint-run-robot-eval-job` is the job-level headless workflow around a
robot-team request. It reads a local request manifest containing customer,
site/package, task/scenario, robot profile, the six robot-team submission
modalities, operation, simulator preference, Cosmos/training preference, budget,
rights/privacy scope, owner system, provenance, and timestamp alignment.

The orchestrator validates rights/privacy and policy evidence, builds or
requires robot-eval dataset cards, runs gated SDK operator planning, writes GPU
provisioning request/result manifests, runs only allowed fixture or command
simulator paths, writes training request/result manifests, copies Site Eval
Director normalization/calibration/breakage artifacts when fixture evaluation is
used, and writes a final job run manifest plus blocked manifest when any gate or
proof requirement fails.

Agents can choose next deterministic commands, inspect manifests/logs, trigger
allowed deterministic reruns, request provisioning, summarize blockers, and
route human review.
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
- one or more `--allow-simulator <isaac_sim|isaac_lab_arena|mujoco|pybullet|newton>`
- explicit `--simulator-command <framework>=<command>`

Optional local CPU preflight smoke requires both:

- `BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true`
- `--allow-cpu-simulator-preflight`

Rendering is not required. PyBullet TinyRenderer is attempted only with
`--allow-cpu-preflight-render`; MuJoCo rendering is not attempted by this lane.

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

Live Agents SDK operators require:

- `OPENAI_API_KEY`
- `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`
- `--allow-live-agent-operator` for `blueprint-run-simulation-automation` and
  `blueprint-run-robot-eval-job`
- `--allow-live-agents-sdk-operator` for `blueprint-run-site-eval-director`
- `--agent-mode agents-sdk` where the command uses an agent mode switch
- optional `BLUEPRINT_ALLOW_AGENT_EXTERNAL_ACTIONS=true` for external actions
- optional `BLUEPRINT_ALLOW_AGENT_SPEND_ACTIONS=true` for spend-bearing actions

Live Codex code-maintainer operators require:

- `BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true`
- either `OPENAI_API_KEY` plus a Python Codex SDK dependency, or an installed
  authenticated `codex` CLI plus `BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true`
- `--allow-live-agent-operator` for `blueprint-run-simulation-automation --agent-mode codex-sdk`
- `--allow-live-codex-sdk-operator` for `blueprint-run-site-eval-director --codex-sdk-code-maintainer`
- optional `BLUEPRINT_ALLOW_AGENT_EXTERNAL_ACTIONS=true` for external actions
- optional `BLUEPRINT_ALLOW_AGENT_SPEND_ACTIONS=true` for spend-bearing actions

All live SDK operator manifests record decisions, tool-call summaries, commands
chosen, refusals, blockers, and proof effect. Agents may never directly set
proof booleans true without deterministic accepted artifacts.

## Proof Boundaries

### Local Review Artifacts

Local review artifacts include simready, Marble handoff, robot-eval dataset, and
simulation automation manifests. They can support planning and review only.

They do not prove simulator execution, physics/contact validity, robot policy
success, safety validation, training completion, or public deployment readiness.

### Simulator Execution Proof

Simulator execution proof requires a successful owner proof package with command,
stdout, stderr, exit code, scene load trace, spawn trace, action/policy trace,
artifact manifest, pass/fail criteria, and owner attestation from the owner
simulator. The validator rejects proof packages that try to mark robot readiness,
policy success, safety, public claim upgrades, or real robot contact as proven.

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

CPU/pre-GPU setup from an explicit local scene asset:

```bash
blueprint-run-simulation-automation \
  --capture-root /path/to/capture-root \
  --scene-asset /path/to/local-scene.ply
```

Optional local CPU smoke, if dependencies are installed:

```bash
BLUEPRINT_ALLOW_CPU_SIMULATOR_PREFLIGHT=true \
blueprint-run-cpu-simulator-preflight \
  --capture-root /path/to/capture-root \
  --allow-cpu-simulator-preflight \
  --backend pybullet \
  --backend mujoco
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

Optional gated live SDK operators:

```bash
BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true \
blueprint-run-site-eval-director \
  --capture-root /path/to/capture-root \
  --agents-sdk-site-eval \
  --allow-live-agents-sdk-operator

BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS=true \
blueprint-run-site-eval-director \
  --capture-root /path/to/capture-root \
  --codex-sdk-code-maintainer \
  --allow-live-codex-sdk-operator
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

Post-Training Data Package export and archive:

```bash
blueprint-build-post-training-data-package \
  --capture-root /path/to/capture-root \
  --job-dir /path/to/capture-root/pipeline/robot_eval_jobs/<job_id>
```

Arena result ingest and package lane:

```bash
blueprint-ingest-arena-results \
  --capture-root /path/to/capture-root \
  --arena-results-dir /path/to/isaac-lab-arena-results \
  --scenario-count 500 \
  --shard-size 50
```

Optional OpenAI rollout vision labels:

```bash
BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true \
blueprint-ingest-arena-results \
  --capture-root /path/to/capture-root \
  --arena-results-dir /path/to/isaac-lab-arena-results \
  --allow-rollout-vision-labeling \
  --vision-labeling-command "blueprint-label-rollout-vision-openai --output-dir ."
```

The OpenAI hook writes `rollout_vision_labels.command.json`. Ingest consumes
those labels as review-required support evidence; they do not prove contact,
safety, policy execution, or robot readiness.

Arena package proof-boundary audit:

```bash
blueprint-audit-arena-package \
  --capture-root /path/to/capture-root \
  --package-dir /path/to/capture-root/pipeline/robot_eval_jobs/<job_id> \
  --expected-scenario-count 500 \
  --require-job-artifacts
```

Live setup and external-gate preflight:

```bash
blueprint-audit-live-pipeline-setup \
  --capture-root /path/to/capture-root \
  --package-dir /path/to/capture-root/pipeline/robot_eval_jobs/<job_id> \
  --digitalocean-droplet-name paperclip-prod-01 \
  --digitalocean-droplet-ip 206.81.11.69
```

This preflight reports whether the local machine has the live gates, command
hooks, Codex CLI, SDK modules, package audit, and WebApp upstream IDs required
for a live run. It may treat a 24/7 droplet as a control-plane target, but not
as simulator, contact, safety, or robot-readiness proof. ChatGPT Pro/Codex OAuth
can be used through an authenticated `codex` CLI only when
`BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true` and the matching live Codex gate are
set; otherwise repo subprocesses still require explicit API keys or
OAuth-owning command hooks.

`--allow-rollout-vision-labeling`, `--allow-delivery-upload`,
`--operator-mode agents-sdk`, `--allow-live-agents-sdk`, and
`--allow-live-codex-sdk` still require their matching environment gates. The
local `--operator-mode fake` path requires `BLUEPRINT_ALLOW_FAKE_LIVE_OPERATORS`
and only proves local operator-control code paths.

Site/capture batch registry with retry/resume status:

```bash
blueprint-build-capture-batch-registry \
  --capture-root /path/to/capture-root \
  --registry-path /path/to/site_capture_batch_registry.json \
  --retry-stage gpu_handoff
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
