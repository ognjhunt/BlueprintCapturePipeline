# BlueprintCapturePipeline

`BlueprintCapturePipeline` is the packaging, trust, and runtime service that turns raw Blueprint captures into real-site robot evaluation artifacts and Post-Training Data Package artifacts with provenance, privacy, and rights safety. World-model, generated, simulation, editing, and augmentation outputs remain support artifacts inside those packages unless a downstream contract explicitly labels them otherwise.

The current active process is: `BlueprintCapture` output -> privacy-safe World Labs input prep -> World Labs API upload/request -> persisted provider manifests -> materialized World Labs output assets with checksums -> CPU/pre-GPU scene and episode preflight -> simulation automation manifest -> explicitly gated simulator runs. Older scene-memory, retrieval/alignment, Cosmos, single-VM GPU, SimReady, and Marble bridge lanes are legacy/advisory support paths unless a command or artifact explicitly requests them.

For public language, Google/Meta smart glasses are supported only for approved repeat walkthroughs where the assignment, hardware, launch proof, and downstream capture/package proof exist. This repo treats glasses outputs as partial/internal until that proof chain exists.

AI and engineer orientation maps live under [`docs/architecture/`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture):

- [`ai-onboarding-map.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/ai-onboarding-map.md)
- [`source-of-truth-map.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/source-of-truth-map.md)
- [`command-safety-matrix.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/command-safety-matrix.md)
- [`refactor-hotspots.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/architecture/refactor-hotspots.md)

## Scope

Primary product path:

- raw capture materialization from `BlueprintCapture`
- Gemini-backed multimodal capture review
- capture evidence analysis and agent review
- deterministic QA aggregation and trust/provenance assembly
- robot-evaluation/data-package fit scoring and capturer payout recommendation
- optional provider preview routing
- privacy-safe World Labs input preparation
- World Labs upload/request/operation/world manifest persistence
- World Labs output asset materialization into local checksum/provenance manifests
- webapp sync for buyer-review surfaces
- Site Cards, Task Cards, Scenario Cards, Eval Cards, rights packets, and proof boundaries
- Post-Training Data Package artifacts such as curated clip/label/export support
- CPU/pre-GPU scene asset inspection, episode specs, and simulator preflight setup
- fail-closed simulation automation manifests
- deterministic object indexing and scene semantics when deeper work is requested
- optional legacy scene-memory assembly
- optional legacy presentation-world assembly
- optional evaluation-prep packaging
- optional legacy runtime registration support for the built site-world package

Support / trust alpha artifacts:

- `qualification_summary.json`
- `capture_quality_summary.json`
- `rights_and_compliance_summary.json`
- `buyer_trust_score.json`
- `world_model_fit_summary.json`
- `capturer_payout_recommendation.json`
- `recapture_requirements.json`
- `provider_preview_status.json`
- `provenance_summary.json`
- `gemini_capture_fidelity_review.json`
- `provider_preview_qa_manifest.json`
- `production_handoff_readiness_manifest.json`

Artifact families and advisory downstream outputs:

- `scene_memory/*`
- `presentation_world/presentation_bundle.json`
- `presentation_world/presentation_world_manifest.json`
- `presentation_world/runtime_demo_manifest.json`
- `evaluation_prep/site_world_spec.json`
- `evaluation_prep/site_world_registration.json`
- `evaluation_prep/site_world_health.json`
- `evaluation_prep/evaluation_prep_manifest.json`
- `simready/simready_scene_manifest.json`
- `simready/isaac_sim/site_scene.usda`
- `simready/mujoco/site_scene.xml`
- `simready/pybullet/site_scene.urdf`
- `marble_sim_assets/marble_asset_manifest.json`
- `marble_sim_assets/marble_simready_bridge.json`
- `robot_eval_dataset/robot_eval_dataset_manifest.json`
- `robot_eval_dataset/real_site_robot_eval_dataset_manifest.json`
- `robot_eval_dataset/site_card.json`
- `robot_eval_dataset/task_cards.json`
- `robot_eval_dataset/scenario_cards.json`
- `robot_eval_dataset/eval_cards.json`
- `robot_eval_dataset/annotation_backlog.json`
- `robot_eval_dataset/proof_boundaries.json`
- `robot_eval_dataset/rights_packet.json`
- `robot_eval_dataset/rights_ledger.json`
- `robot_eval_dataset/task_ontology_v1.json`
- `robot_eval_dataset/scenario_family_library.json`
- `robot_eval_dataset/scoring_methodology.json`
- `robot_eval_dataset/task_thresholds.json`
- `robot_eval_dataset/publication_readiness.json`
- `robot_eval_dataset/recorded_trace_eval_report.json`
- `robot_eval_dataset/policy_eval_report.json`
- `robot_eval_dataset/prediction_outcome_ledger.json`
- `robot_eval_dataset/prediction_vs_actual_summary.json`
- `simulation_automation/simulation_automation_plan.json`
- `simulation_automation/simulation_automation_run_manifest.json`
- `simulation_automation/scene_asset_inventory.json`
- `simulation_automation/scene_asset_dependency_audit.json`
- `simulation_automation/scene_asset_preflight.json`
- `simulation_automation/scene_asset_inspection.json`
- `simulation_automation/scene_frame_estimate.json`
- `simulation_automation/collider_proxy_plan.json`
- `simulation_automation/cpu_scene_proxy_manifest.json`
- `simulation_automation/cpu_preflight_scorecard.json`
- `simulation_automation/task_anchor_proposal_manifest.json`
- `simulation_automation/episode_spec.v1.json`
- `simulation_automation/episode_specs.json`
- `simulation_automation/episode_spec_manifest.json`
- `simulation_automation/agent_episode_spec_proposals.json`
- `simulation_automation/episode_setup_manifest.json`
- `simulation_automation/spawn_pose_validation_manifest.json`
- `simulation_automation/cpu_preflight_manifest.json`
- `simulation_automation/pre_gpu_readiness_summary.json`
- `simulation_automation/cpu_simulator_preflight_manifest.json`
- `simulation_automation/gpu_handoff_packet.json`
- `simulation_automation/gpu_owner_system_proof_schema.json`
- `simulation_automation/owner_gpu_simulator_execution_proof_manifest.json` when
  owner proof is supplied and accepted
- `simulation_automation/gpu_run_checklist.md`
- `simulation_automation/owner_gpu_simulator_execution_blocked_manifest.json`
- `simulation_automation/mujoco_cpu_preflight/*`
- `simulation_automation/pybullet_cpu_preflight/*`
- `simulation_automation/asset_conversion_plan.json`
- `simulation_automation/simulator_execution_manifest.json`
- `simulation_automation/training_orchestration_manifest.json`
- `simulation_automation/proof_boundary.json`
- `simulation_automation/agent_decision_ledger.json`
- `simulation_automation/scenario_execution_plan.json`
- `simulation_automation/task_simulation_requests.json`
- `simulation_automation/scenario_simulator_matrix.json`
- `simulation_automation/agent_review_queue.json`
- `simulation_automation/site_eval_director_run_manifest.json`
- `simulation_automation/site_eval_director_proof_boundary.json`
- `robot_eval_jobs/<job_id>/job_request.json`
- `robot_eval_jobs/<job_id>/job_validation.json`
- `robot_eval_jobs/<job_id>/job_plan.json`
- `robot_eval_jobs/<job_id>/agent_orchestration_plan.json`
- `robot_eval_jobs/<job_id>/gpu_provisioning_request.json`
- `robot_eval_jobs/<job_id>/gpu_provisioning_result.json`
- `robot_eval_jobs/<job_id>/simulator_service_request.json`
- `robot_eval_jobs/<job_id>/simulator_service_result.json`
- `robot_eval_jobs/<job_id>/policy_package_manifest.json`
- `robot_eval_jobs/<job_id>/training_request.json`
- `robot_eval_jobs/<job_id>/training_result.json`
- `robot_eval_jobs/<job_id>/evaluation_request.json`
- `robot_eval_jobs/<job_id>/evaluation_result.json`
- `robot_eval_jobs/<job_id>/normalized_attempt_trace.json`
- `robot_eval_jobs/<job_id>/failure_labels.json`
- `robot_eval_jobs/<job_id>/prediction_outcome_ledger.json`
- `robot_eval_jobs/<job_id>/calibration_report.json`
- `robot_eval_jobs/<job_id>/breakage_library.json`
- `robot_eval_jobs/<job_id>/post_training_data_package_export_manifest.json`
- `robot_eval_jobs/<job_id>/proof_boundary.json`
- `robot_eval_jobs/<job_id>/job_run_manifest.json`
- `robot_eval_jobs/<job_id>/blocked_manifest.json` when blocked
- `robot_eval_job_requests/inbox_run_manifest.json` when a request inbox is consumed
- `site_capture_batch_registry.json` when the capture batch registry command is
  pointed at a registry path

## Local Development

```bash
uv sync --extra dev
```

This is a repository development setup only. It is not the supported single-VM GPU runtime bootstrap path.

Optional LLM support for the capture review agent:

```bash
uv sync --extra dev --extra llm
```

Local tests automatically add `src/` and the sibling `BlueprintContracts/src` to `sys.path` through [`tests/conftest.py`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/tests/conftest.py). If the contracts repo is not present beside this repo, install `blueprint-contracts` before running `pytest`.

Cross-repo external alpha gate:

```bash
python scripts/run_external_alpha_launch_gate.py
```

## Privacy And World Labs Input

The current World Labs preview path requires a production-audited
`privacy/final_walkthrough.*` or audited derivative before provider upload.
SAM3, VIP/depth, and DeepPrivacy2 can be configured as optional HTTP or command
runner hooks, but the production gate is the final walkthrough audit rather than
proof that those exact model backends ran.

- `PRIVACY_SAM3_URL`
- `PRIVACY_VIP_URL`
- `PRIVACY_DEPTH_ANYTHING_URL` (optional; otherwise `vip-inpaint` handles depth-only requests)
- `PRIVACY_DEEPPRIVACY2_URL`
- `PRIVACY_RUNNER_TOKEN`
- `PRIVACY_SAM3_COMMAND`
- `PRIVACY_VIP_COMMAND`
- `PRIVACY_DEPTH_ANYTHING_COMMAND`
- `PRIVACY_DEEPPRIVACY2_COMMAND`

Production preview packets can be checked locally before provider submission:

```bash
blueprint-validate-provider-preview-packet \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --mode production \
  --require-webapp-sync
```

The validator writes `pipeline/provider_preview_qa_manifest.json`. In production
mode, raw-video bypass, missing privacy verification, missing input checksums,
missing or placeholder WebApp upstream ids, or mismatched
canonical/provider-adapter input URIs block provider-ready status.

After World Labs manifests, materialized assets, Marble handoff, CPU preflight,
and GPU handoff artifacts exist, build the final handoff summary:

```bash
blueprint-build-production-handoff-readiness \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --mode production
```

That command writes `pipeline/production_handoff_readiness_manifest.json`.
`ready_except_owner_gpu_simulator_execution` means the repo-local handoff packet
is complete, production WebApp upstream-link truth is present, and the only
remaining unproven step is owner-system simulator execution. It still does not
prove robot readiness.

For temporary internal demos, `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` allows the World Labs preview path to fall back to the raw walkthrough video when privacy processing is unavailable. The bypass path is intentionally labeled as non-production and unredacted, and the input video is auto-trimmed/compressed to World Labs upload limits before submission.

The production privacy deployment may use the privacy runner services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`

Legacy `video_to_world`, retrieval-index, and live-geometry validation material
remains in older docs for compatibility, but those paths are not part of the
active Capture App -> World Labs -> CPU preflight -> simulation-manifest flow.

The main `blueprint-pipeline` job stays CPU-only. The concrete service contract, storage behavior, and model-path rules are documented in [docs/PRIVACY_RUNNER_SERVICES.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/PRIVACY_RUNNER_SERVICES.md).

The privacy path treats depth generation as a first-class optional artifact when
depth evidence is available or a depth runner is configured:

- use ARKit depth/confidence when available
- otherwise run Depth Anything 3 only when the depth runner is configured for the lane
- persist the resulting depth and confidence manifests for downstream grounding
- pass those manifests into VIP so non-ARKit inpainting reuses the generated depth artifacts

## Legacy GPU Bring-Up

The older single-VM GPU runbook is still available for legacy downstream world-model work in [docs/GPU_VM_RUNBOOK.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/GPU_VM_RUNBOOK.md), but it is not the active preview, upload, CPU-preflight, or simulation-manifest path.

For privacy-service bring-up, use the service images under [`deploy/docker/`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/docker) and the Terraform stack under [`deploy/terraform/main.tf`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/terraform/main.tf).

The normal local repo bootstrap is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

Then stage and run:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /data/raw_bundle \
  --storage-root /data/blueprint-storage \
  --bucket local-blueprint \
  --copy \
  --run-qualification \
  --pipeline-lane current
```

## Entry Points

Current pipeline:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane current
```

`current` and `all` expand to qualification, evaluation prep, and simulation
automation. World Labs API submission happens inside qualification only when the
descriptor requests `preview_simulation` or `preview` and the privacy-safe World
Labs input is ready.

Raw bundle staging:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /path/to/raw-download-folder \
  --storage-root /mnt/blueprint-storage \
  --bucket local-blueprint \
  --link \
  --run-qualification \
  --pipeline-lane current
```

Qualification agent review:

```bash
blueprint-agent-review \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider openai
```

Optional agent-review wrapper:

```bash
blueprint-run-e2e \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider openai
```

Explicit legacy scene-memory build:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane scene_memory
```

Legacy local staging lanes can still be requested explicitly through
`scripts/stage_capture_bundle.py` with `--pipeline-lane scene_memory`,
`retrieval_index`, `frame_alignment`, `synthesis_coverage_validation`, or
`cosmos_single_capture_smoke` when `--run-qualification` is set. These lanes
still honor geometry/provider truth and will not promote fallback geometry into
live `video_to_world`, simulator, or robot-readiness proof.

Object index build:

```bash
blueprint-build-object-index \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Evaluation prep build:

```bash
blueprint-build-evaluation-prep \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider manual
```

Legacy local simulator-review artifact module:

```bash
PYTHONPATH=src python -m blueprint_pipeline.simready_assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

The simready asset lane is documented in
[`docs/SIMREADY_ASSET_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/SIMREADY_ASSET_LANE.md).
It writes review artifacts only; it does not run Isaac Sim, MuJoCo, PyBullet,
live providers, model downloads, or robot-readiness trials.
Evaluation prep surfaces existing SimReady artifacts but does not auto-build
them unless `BLUEPRINT_ALLOW_LEGACY_SIMREADY_EVAL_PREP=true` is set.

Legacy local Marble sim-asset handoff module:

```bash
PYTHONPATH=src python -m blueprint_pipeline.marble_sim_assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Use `--world-manifest /path/to/worldlabs_world_manifest.json` to review an
explicit local World Labs world manifest. The Marble handoff lane is documented
in
[`docs/MARBLE_SIM_ASSET_HANDOFF.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/MARBLE_SIM_ASSET_HANDOFF.md).
It reads persisted World Labs manifests and emits Isaac Sim, MuJoCo, and
PyBullet review packets without downloading remote assets, calling World Labs,
running simulators, or claiming robot readiness.
Evaluation prep surfaces existing Marble bridge artifacts but does not
auto-build them unless `BLUEPRINT_ALLOW_LEGACY_MARBLE_EVAL_PREP=true` is set.

World Labs output asset materialization:

```bash
blueprint-materialize-worldlabs-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

This downloads already-generated Marble asset URLs, by default only the
collider GLB needed for CPU/pre-GPU handoff, into `pipeline/worldlabs_assets/`
and writes `pipeline/worldlabs_export_manifest.json` with checksums and source
URLs. It does not start a new World Labs generation, run simulators, or prove
robot readiness.

Fail-closed simulation automation plan:

```bash
blueprint-run-simulation-automation \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

The simulation automation lane is documented in
[`docs/SIMULATION_AUTOMATION_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/SIMULATION_AUTOMATION_LANE.md).
It writes local orchestration manifests only. It does not run simulators,
download assets, start training, call providers, or prove robot readiness unless
explicit per-run approvals and dependencies are present.

Optional deterministic site-eval director plan:

```bash
blueprint-run-site-eval-director \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

The site-eval director reads the local robot-eval Site, Task, Scenario, Eval,
and proof-boundary cards plus existing World Labs, Marble, simready, and
simulation automation manifests. It writes local scenario execution plans, task
simulation request manifests, simulator matrices, fixture-backed normalized
attempt traces, failure labels, updated Eval Card views, prediction/outcome
ledgers, calibration reports, breakage libraries, Cosmos export/request
manifests, review queues, and proof boundaries under
`pipeline/simulation_automation/`. Fixture attempts prove only the local
deterministic loop; real simulator, robot, safety, training, and public-claim
upgrades remain blocked without owner-system proof and explicit gates. Optional
`--agents-sdk-site-eval` and `--codex-sdk-code-maintainer` flags only write
advisory SDK request or blocked manifests; they do not run agents, simulators,
providers, downloads, training, deployments, payments, or proof upgrades.

Optional headless robot-eval job orchestration:

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-request /path/to/robot-eval-job-request.json \
  --job-id <job_id> \
  --agent-mode fake \
  --provisioner fixture_local \
  --simulator fixture
```

To consume WebApp-exported request JSON files, point the same entrypoint at an
inbox:

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-request-inbox /path/to/robot-eval-job-request-inbox \
  --provisioner fixture_local \
  --simulator fixture
```

The job orchestrator reads a robot-team request for policy/container/trace/demo
references, robot profile, task/scenario scope, rights/privacy scope, operation,
simulator preference, training preference, budget, owner system, provenance, and
timestamp alignment. It validates the request, writes a deterministic state
machine under `pipeline/robot_eval_jobs/<job_id>/`, invokes fixture/local
surfaces when allowed, and writes exact blocked manifests for missing evidence
or denied gates. The inbox runner also copies each accepted request under
`pipeline/robot_eval_job_requests/<job_id>/job_request.json` and writes
`pipeline/robot_eval_job_requests/inbox_run_manifest.json`. Fixture provisioner
and fixture simulator paths prove only the repo-local orchestration loop. Vast, RunPod, GCP, local process, Docker,
MuJoCo, PyBullet, Newton, Isaac Sim, Agents SDK, and Cosmos training paths stay
blocked unless their explicit environment and CLI gates are present.

Post-Training Data Package export manifest:

```bash
blueprint-build-post-training-data-package \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id>
```

Site/capture batch registry with retry/resume status:

```bash
blueprint-build-capture-batch-registry \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --registry-path /path/to/site_capture_batch_registry.json \
  --retry-stage gpu_handoff
```

The registry tracks privacy, World Labs, materialization, CPU preflight, GPU
handoff, eval result, and data-package export status per site/capture. It does
not perform the stages itself or upgrade readiness booleans.

## Contract Boundary

Shared contract code lives in `BlueprintContracts`:

- `handoff_contract`
- `site_world_contract`
- `runtime_layer_contract`
- `canonical_package`

The bridge contract for this repo is documented in [`docs/CAPTURE_BRIDGE_CONTRACT.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/CAPTURE_BRIDGE_CONTRACT.md).

Current cross-repo implementation status is tracked in [`docs/READINESS_MATRIX.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/READINESS_MATRIX.md). It is intentionally strict about what is shipped in-repo versus what still depends on live GPU/runtime/model access.
