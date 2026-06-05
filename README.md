# BlueprintCapturePipeline

`BlueprintCapturePipeline` is the packaging, trust, and runtime service that turns raw Blueprint captures into real-site robot evaluation artifacts and Post-Training Data Package artifacts with provenance, privacy, and rights safety. World-model, site-world, generated, simulation, editing, and augmentation outputs remain support artifacts inside those packages unless a downstream contract explicitly labels them otherwise.

For non-ARKit captures, the canonical support-artifact path remains internal: `BlueprintCapture` evidence -> support/trust analysis -> privacy-aware geometry staging -> retrieval memory -> alignment -> synthesis/Cosmos conditioning. The default hosted preview path is now World Labs Marble from the privacy-safe walkthrough, while scene-memory, presentation, evaluation-prep, generated-data, and runtime registration remain downstream derived lanes.

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
- webapp sync for buyer-review surfaces
- Site Cards, Task Cards, Scenario Cards, Eval Cards, rights packets, and proof boundaries
- Post-Training Data Package artifacts such as curated clip/label/export support
- deterministic object indexing and scene semantics when deeper work is requested
- optional scene-memory assembly
- optional presentation-world assembly
- optional evaluation-prep packaging
- optional runtime registration support for the built site-world package

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

Optional legacy downstream artifacts:

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
- `robot_eval_dataset/recorded_trace_eval_report.json`
- `robot_eval_dataset/policy_eval_report.json`
- `robot_eval_dataset/prediction_outcome_ledger.json`
- `robot_eval_dataset/prediction_vs_actual_summary.json`
- `simulation_automation/simulation_automation_plan.json`
- `simulation_automation/simulation_automation_run_manifest.json`
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
- `robot_eval_jobs/<job_id>/proof_boundary.json`
- `robot_eval_jobs/<job_id>/job_run_manifest.json`
- `robot_eval_jobs/<job_id>/blocked_manifest.json` when blocked

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

## Privacy Runner Services

The production preview path expects URL-first privacy runners:

- `PRIVACY_SAM3_URL`
- `PRIVACY_VIP_URL`
- `PRIVACY_DEPTH_ANYTHING_URL` (optional; otherwise `vip-inpaint` handles depth-only requests)
- `PRIVACY_DEEPPRIVACY2_URL`
- `PRIVACY_RUNNER_TOKEN`

For temporary internal demos, `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` allows the World Labs preview path to fall back to the raw walkthrough video when privacy processing is unavailable. The bypass path is intentionally labeled as non-production and unredacted, and the input video is auto-trimmed/compressed to World Labs upload limits before submission.

The non-ARKit geometry path expects a dedicated GPU `video_to_world` runner. This
is the only path that can mark non-ARKit geometry as live world-model-ready:

- `VIDEO_TO_WORLD_URL`
- `VIDEO_TO_WORLD_RUNNER_TOKEN`
- `VIDEO_TO_WORLD_PIPELINE_PRESET` or `VIDEO_TO_WORLD_COMMAND_TEMPLATE`

The production deployment should use four GPU Cloud Run services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`
- `video-to-world`

Recommended `video_to_world` presets:

- `preprocess_only` for DA3-only geometry bootstrap
- `preprocess_plus_alignment` for DA3 + non-rigid alignment outputs. This is the default deployment preset.
- `full_fast` for the end-to-end upstream reconstruction path with the lighter preset
- `full_extensive` for the full upstream path including global optimization and longer inverse-deformation / GS stages

If the runner is missing or fails, the geometry stage may write an explicitly
labeled internal fallback so local tests and contract-shape debugging can continue.
Fallback geometry is machine-readable as `geometry_source=fallback_geometry` and
`fallback_used=true`; it must remain `ready_for_world_model=false`,
`geometry_live_ready=false`, and `site_faithful_market_ready=false`.

`RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO=true` is now the default production expectation. Retrieval indexing fails closed unless it can resolve `world_model_video_uri`, `privacy_processed_video_uri`, or the concrete privacy artifact at `privacy/final_walkthrough.mov` / `privacy/final_walkthrough.mp4`.

The main `blueprint-pipeline` job stays CPU-only. The concrete service contract, storage behavior, and model-path rules are documented in [docs/PRIVACY_RUNNER_SERVICES.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/PRIVACY_RUNNER_SERVICES.md).

Live geometry validation command:

```bash
VIDEO_TO_WORLD_URL=https://<video-to-world-runner> \
VIDEO_TO_WORLD_RUNNER_TOKEN=<secret> \
VIDEO_TO_WORLD_PIPELINE_PRESET=preprocess_plus_alignment \
python3 scripts/run_geometry_lane.py \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider video_to_world \
  --model video_to_world-default
```

Before claiming live proof, inspect `pipeline/geometry/geometry_summary.json` and
`pipeline/geometry/logs/provider_result.json` for `geometry_source=video_to_world`,
`fallback_used=false`, `provider_native_result=true`, `ready_for_world_model=true`,
and `geometry_live_ready=true`.

The privacy path now treats depth generation as a first-class step:

- use ARKit depth/confidence when available
- otherwise run Depth Anything 3 for every non-ARKit capture, including glasses captures, even if no humans are detected
- persist the resulting depth and confidence manifests for downstream grounding
- pass those manifests into VIP so non-ARKit inpainting reuses the generated depth artifacts

## Local GPU Bring-Up

The older single-VM GPU runbook is still available for legacy downstream world-model work in [docs/GPU_VM_RUNBOOK.md](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/GPU_VM_RUNBOOK.md), but it is not the active preview path.

For privacy-service bring-up, use the service images under [`deploy/docker/`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/docker) and the Terraform stack under [`deploy/terraform/main.tf`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/deploy/terraform/main.tf).

The local repo bootstrap remains:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_ml_stack.sh
python3 scripts/setup_environment.py --check
```

Then stage and run:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /data/raw_bundle \
  --storage-root /data/blueprint-storage \
  --bucket local-blueprint \
  --copy \
  --run-qualification
```

## Entry Points

Pipeline lanes:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane qualification
```

Raw bundle staging:

```bash
python3 scripts/stage_capture_bundle.py \
  --source-bundle /path/to/raw-download-folder \
  --storage-root /mnt/blueprint-storage \
  --bucket local-blueprint \
  --link \
  --run-qualification
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

Optional legacy scene-memory build:

```bash
blueprint-capture-pipeline \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json \
  --lane scene_memory
```

Deeper local staging lanes can be requested through `scripts/stage_capture_bundle.py`
with `--pipeline-lane retrieval_index`, `frame_alignment`, `evaluation_prep`,
`synthesis_coverage_validation`, `cosmos_single_capture_smoke`, or `all` when
`--run-qualification` is set. These lanes still honor geometry/provider truth
and will not promote fallback geometry into live `video_to_world` proof.

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

Optional local simulator-review artifact build:

```bash
blueprint-build-simready-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

The simready asset lane is documented in
[`docs/SIMREADY_ASSET_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/SIMREADY_ASSET_LANE.md).
It writes review artifacts only; it does not run Isaac Sim, MuJoCo, PyBullet,
live providers, model downloads, or robot-readiness trials.

Optional local Marble sim-asset handoff build:

```bash
blueprint-build-marble-sim-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Use `--world-manifest /path/to/worldlabs_world_manifest.json` to review an
explicit local World Labs world manifest. The Marble handoff lane is documented
in
[`docs/MARBLE_SIM_ASSET_HANDOFF.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/MARBLE_SIM_ASSET_HANDOFF.md).
It reads persisted World Labs manifests and emits Isaac Sim, MuJoCo, and
PyBullet review packets without downloading remote assets, calling World Labs,
running simulators, or claiming robot readiness.

Optional fail-closed simulation automation plan:

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

The job orchestrator reads a robot-team request for policy/container/trace/demo
references, robot profile, task/scenario scope, rights/privacy scope, operation,
simulator preference, training preference, budget, owner system, provenance, and
timestamp alignment. It validates the request, writes a deterministic state
machine under `pipeline/robot_eval_jobs/<job_id>/`, invokes fixture/local
surfaces when allowed, and writes exact blocked manifests for missing evidence
or denied gates. Fixture provisioner and fixture simulator paths prove only the
repo-local orchestration loop. Vast, RunPod, GCP, local process, Docker,
MuJoCo, PyBullet, Newton, Isaac Sim, Agents SDK, and Cosmos training paths stay
blocked unless their explicit environment and CLI gates are present.

## Contract Boundary

Shared contract code lives in `BlueprintContracts`:

- `handoff_contract`
- `site_world_contract`
- `runtime_layer_contract`
- `canonical_package`

The bridge contract for this repo is documented in [`docs/CAPTURE_BRIDGE_CONTRACT.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/CAPTURE_BRIDGE_CONTRACT.md).

Current cross-repo implementation status is tracked in [`docs/READINESS_MATRIX.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/READINESS_MATRIX.md). It is intentionally strict about what is shipped in-repo versus what still depends on live GPU/runtime/model access.
