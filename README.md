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
- `palatial_physready/twin_candidate_manifest.json`
- `palatial_physready/palatial_request_manifest.json`
- `palatial_physready/palatial_physready_run_manifest.json`
- `palatial_physready/materialization_manifest.json`
- `palatial_physready/validation_manifest.json`
- `palatial_physready/assets/*`
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
- `simulation_automation/scenario_variation_instances.json`
- `simulation_automation/arena_environment_packet.json`
- `simulation_automation/simulator_engine_plugin_registry.json`
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
- `robot_eval_jobs/<job_id>/scheduler_decision.json`
- `robot_eval_jobs/<job_id>/worker_launch_plan.json`
- `robot_eval_jobs/<job_id>/worker_manifest.json`
- `robot_eval_jobs/<job_id>/gpu_provisioning_request.json`
- `robot_eval_jobs/<job_id>/gpu_provider_launch_request.json`
- `robot_eval_jobs/<job_id>/gpu_provider_launcher_result.json` when
  `blueprint-run-gpu-provider-launcher` is run
- `robot_eval_jobs/<job_id>/gpu_provider_launcher.stdout.log` when
  `blueprint-run-gpu-provider-launcher` is run
- `robot_eval_jobs/<job_id>/gpu_provider_launcher.stderr.log` when
  `blueprint-run-gpu-provider-launcher` is run
- `robot_eval_jobs/<job_id>/runpod_provider_adapter_result.json` when
  `blueprint-run-runpod-provider-adapter` is run
- `robot_eval_jobs/<job_id>/gpu_cost_control_ledger.json`
- `robot_eval_jobs/<job_id>/gpu_provisioning_result.json`
- `robot_eval_jobs/<job_id>/simulator_service_request.json`
- `robot_eval_jobs/<job_id>/simulator_service_result.json`
- `robot_eval_jobs/<job_id>/scenario_eval_matrix.json`
- `robot_eval_jobs/<job_id>/policy_package_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_observation_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_observations.jsonl`
- `robot_eval_jobs/<job_id>/robot_pov_frame_sequence_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_render_storyboard.json`
- `robot_eval_jobs/<job_id>/policy_execution_manifest.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.jsonl`
- `robot_eval_jobs/<job_id>/policy_adapter_manifest.json` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/training_request.json`
- `robot_eval_jobs/<job_id>/training_result.json`
- `robot_eval_jobs/<job_id>/evaluation_request.json`
- `robot_eval_jobs/<job_id>/evaluation_result.json`
- `robot_eval_jobs/<job_id>/arena_eval_schedule.json` when Arena package ingest
  is run
- `robot_eval_jobs/<job_id>/arena_result_ingest_ledger.json` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/arena_eval_metrics.json` when Arena package ingest
  is run
- `robot_eval_jobs/<job_id>/normalized_attempt_trace.json`
- `robot_eval_jobs/<job_id>/failure_labels.json`
- `robot_eval_jobs/<job_id>/clips_manifest.json` when Arena package ingest is run
- `robot_eval_jobs/<job_id>/rollout_vision_labels.json` when Arena package ingest
  is run
- `robot_eval_jobs/<job_id>/review_resolution_ledger.json` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/accepted_failure_labels.json` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/prediction_outcome_ledger.json`
- `robot_eval_jobs/<job_id>/calibration_report.json`
- `robot_eval_jobs/<job_id>/breakage_library.json`
- `robot_eval_jobs/<job_id>/deployment_outcome_intake_manifest.json`
- `robot_eval_jobs/<job_id>/deployment_outcome_ledger.json`
- `robot_eval_jobs/<job_id>/sim_vs_real_calibration_report.json`
- `robot_eval_jobs/<job_id>/prediction_vs_actual_deployment_summary.json`
- `robot_eval_jobs/<job_id>/real_world_validation_followup_plan.json`
- `robot_eval_jobs/<job_id>/real_world_validation_followup_request_queue.json`
- `robot_eval_jobs/<job_id>/live_eval_closure_manifest.json`
- `robot_eval_jobs/<job_id>/customer_handoff_report.md` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/customer_handoff_report.json` when Arena package
  ingest is run
- `robot_eval_jobs/<job_id>/delivery_manifest.json` when Arena package ingest
  is run
- `robot_eval_jobs/<job_id>/arena_rerun_plan.json` when Arena package ingest is run
- `robot_eval_jobs/<job_id>/live_operator_ledger.json` when Arena package ingest
  is run
- `robot_eval_jobs/<job_id>/dataset_card.json`
- `robot_eval_jobs/<job_id>/license_manifest.json`
- `robot_eval_jobs/<job_id>/package_index.json`
- `robot_eval_jobs/<job_id>/checksums.json`
- `robot_eval_jobs/<job_id>/archive_manifest.json`
- `robot_eval_jobs/<job_id>/post_training_data_package_export_manifest.json`
- `robot_eval_jobs/<job_id>/proof_boundary.json`
- `robot_eval_jobs/<job_id>/startup_architecture_audit.json`
- `robot_eval_jobs/<job_id>/worker_runtime_manifest.json` when run by
  `blueprint-run-robot-eval-worker`
- `robot_eval_jobs/<job_id>/job_run_manifest.json`
- `robot_eval_jobs/<job_id>/blocked_manifest.json` when blocked
- `robot_eval_job_requests/inbox_run_manifest.json` when a request inbox is consumed
- `live_pipeline_setup/live_pipeline_setup_manifest.json` when live setup is audited
- `live_pipeline_control_plane/live_pipeline_control_plane_manifest.json` when the
  always-on control-plane runner is used
- `live_pipeline_control_plane/live_pipeline_external_input_packet.json` and
  `.md` when the always-on control-plane runner publishes the exact external
  inputs still needed
- `live_pipeline_control_plane/live_pipeline_proof_boundary_audit.json` when
  control-plane outputs are audited for internal consistency, missing external
  inputs, secret leakage, and forbidden proof upgrades
- `live_pipeline_control_plane/live_pipeline_input_intake_audit.json` when
  candidate WebApp job requests or owner Arena result directories are validated
  before staging for the control plane
- `live_pipeline_control_plane/live_pipeline_staged_inputs.json` when validated
  WebApp requests or owner Arena result directories are deliberately staged for
  the next control-plane pass
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

Live Arena/package setup audit:

```bash
blueprint-audit-live-pipeline-setup \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Timer-safe control-plane pass for the DigitalOcean droplet:

```bash
blueprint-intake-live-pipeline-inputs \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json \
  --webapp-job-request /path/to/robot_eval_job_request.json \
  --arena-results-dir /path/to/owner-arena-results \
  --policy-package /path/to/robot_team_policy_package.json \
  --real-robot-pov /path/to/real_robot_pov_manifest.json \
  --deployment-outcomes /path/to/deployment_outcome_manifest.json \
  --live-closure-evidence /path/to/live_eval_closure_evidence.json \
  --stage-webapp-request \
  --stage-arena-results \
  --stage-policy-package \
  --stage-real-robot-pov \
  --stage-deployment-outcomes \
  --stage-live-closure-evidence
blueprint-run-live-pipeline-control-plane
blueprint-audit-live-pipeline-proof-boundary \
  --manifest-path /var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json
```

That command audits readiness and optionally drains
`BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX` through the deterministic
`robot_eval_job_request.v1` orchestrator. It writes a blocked/noop manifest plus
`live_pipeline_external_input_packet.json` and `.md` when capture roots, inboxes,
live simulator commands, owner Arena result artifacts, vision-labeling commands,
robot-team policy package references, deployment outcome records, delivery
commands, closure evidence, or live operator credentials are missing. The packet
is a handoff contract only; placeholder WebApp IDs or sample job requests are
never treated as proof. Deployment outcome records can feed prediction-vs-actual
tracking and calibration, but `real_world_outcome_proven` stays false until each
actual outcome record carries owner evidence refs, an owner proof URI, or an
operator/owner attestation. A queued WebApp `robot_eval_job_request.v1` can satisfy
the WebApp upstream-truth
requirement only when it contains `site_submission_id`, `request_id`,
`buyer_request_id`, and `capture_job_id`, its `site_package.capture_root`
matches the configured control-plane capture root, and the request source
identifies the WebApp. Otherwise the closure gate requires those IDs to be
grounded in persisted capture/WebApp handoff artifacts and blocks conflicting
source values.
The proof-boundary audit exits zero for a healthy waiting state and records
remaining external blockers separately from internal artifact or overclaim
failures. It also checks `live_pipeline_staged_inputs.json` when present, so a
bad staged pointer is treated as an internal audit failure rather than a normal
external wait.
The intake command validates candidate handoff files against the configured
capture root and inbox. Add `--stage-webapp-request` only when you want it to
copy a validated WebApp request into the configured inbox; it does not process
the job or run Arena. Add `--stage-arena-results` to write
`live_pipeline_staged_inputs.json`; the next control-plane pass can consume that
validated Arena result directory without an env-file edit. The staged pointer is
still an ingest input only, not simulator execution proof. Add
`--policy-package` plus `--stage-policy-package` to validate and copy a
job-specific robot-team policy handoff into
`pipeline/robot_eval_inputs/<job_id>/policy_package.json`. The job orchestrator
accepts API endpoint, Docker container, recorded action trace, high-level skill
trace, teleop demo, and sim controller plugin modalities, but policy proof still
requires the gated policy execution bundle to produce attempts. The final closure
audit also revalidates selected modality status and required fields, so a
hand-authored manifest cannot pass by naming a modality while leaving its
reference blocked or incomplete. Add
`--real-robot-pov` plus `--stage-real-robot-pov` to validate and copy
owner-supplied robot camera/action evidence to
`pipeline/robot_eval_inputs/real_robot_pov_manifest.json`. Each record must
carry exact `scenario_eval_run_id` and `scenario_variation_instance_id` keys,
camera video, action log, timestamp alignment, and owner evidence or operator
attestation. Generated POV storyboards remain support artifacts only; real POV
proof is allowed only after the robot-eval job ingests matching real robot
evidence for every required scenario eval run. Add
`--deployment-outcomes` plus `--stage-deployment-outcomes` to validate and copy
job-specific actual pilot/deployment records into
`pipeline/robot_eval_inputs/<job_id>/deployment_outcomes/inbox/`; the robot-eval
job still has to pair those records with predictions before sim-vs-real
calibration is proven. Records with task/scenario IDs and actual-result signals
can be staged as real-world validation inputs before proof, but they are only
calibration-ready when each staged record includes `scenario_eval_run_id` or
`scenario_variation_instance_id` for an exact prediction join. Otherwise the
control-plane packet keeps `predicted_vs_actual_exact_match_keys` open. It also
keeps `real_world_deployment_outcome_owner_evidence` open until every staged
record has owner evidence. Add
`--live-closure-evidence` plus `--stage-live-closure-evidence` to validate and
copy job-specific review, delivery, rights/privacy, and safety/contact/physics
evidence into
`pipeline/robot_eval_inputs/<job_id>/live_eval_closure_evidence.json`; the
job-level closure audit is still the only artifact allowed to upgrade readiness.

For live WebApp-to-droplet handoff, run the authenticated intake service:

```bash
BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN=<redacted> \
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765
```

`POST /api/live-pipeline/job-requests` accepts either the direct
`robot_eval_job_request.v1` body or the WebApp queue envelope, validates the
same four WebApp IDs and matching `site_package.capture_root`, stages the file
into `BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX`, and optionally triggers the
control-plane one-shot when `BLUEPRINT_ALLOW_LIVE_PIPELINE_INTAKE_TRIGGER=true`
and `BLUEPRINT_LIVE_PIPELINE_INTAKE_TRIGGER_COMMAND` are set.
`POST /api/live-pipeline/policy-packages` accepts `robot_team_policy_package.v1`
or a direct policy-package body with one supported robot-team modality, validates
the job id and modality-specific required fields, and stages it at
`pipeline/robot_eval_inputs/<job_id>/policy_package.json`.
`POST /api/live-pipeline/real-robot-pov` accepts `real_robot_pov_manifest.v1`,
validates exact run/variation keys plus camera/action evidence refs, and stages
it at `pipeline/robot_eval_inputs/real_robot_pov_manifest.json`.
`POST /api/live-pipeline/deployment-outcomes` accepts
`deployment_outcome_manifest.v1`, `actual_outcome_manifest.v1`, or
`deployment_outcome.v1` JSON, validates job id plus task/scenario/actual-result
fields, audits exact prediction join keys, and stages records under
`pipeline/robot_eval_inputs/<job_id>/deployment_outcomes/inbox/`.
`POST /api/live-pipeline/live-closure-evidence` accepts
`live_robot_eval_closure_evidence.v1`, validates the required review, delivery,
and safety/contact/physics sections, and stages it at
`pipeline/robot_eval_inputs/<job_id>/live_eval_closure_evidence.json`. The
service is an intake layer only; it does not run Arena, set proof booleans, or
publish a claim upgrade.

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

For the current sample-video to owner-GPU proof path, use
[`docs/FIRST_GPU_E2E_RUNBOOK.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/FIRST_GPU_E2E_RUNBOOK.md).
It sequences local capture preflight, current pipeline lanes, WebApp forwarding,
owner GPU simulator command execution, proof ingestion, and closure audits
without promoting CPU or simulator smoke artifacts into robot-readiness proof.
Before staging a loose local video, run
`blueprint-audit-first-gpu-sample-video` to check file existence, suffix, size,
and duration suitability for the first World Labs clip. The staging command can
also enforce the same check with `--require-source-video-preflight`.
The runbook also includes a local WebApp rehearsal request mode that is blocked
by default unless `--allow-local-webapp-rehearsal` is passed, so dry-run request
shape checks cannot be confused with live WebApp forwarding proof.
Use `blueprint-audit-first-gpu-cross-repo-readiness` before the run to audit the
Capture -> Pipeline -> WebApp -> Pipeline source contracts plus the concrete
capture-root readiness gate and generated first-GPU run packet launch order in
one manifest. Its `gpu_spend_decision` is the go/no-go field for RunPod or
equivalent GPU VM allocation, and it remains blocked if the packet is missing,
`first_gpu_webapp_handoff` is blocked, `first_gpu_scene_asset_acquisition` is
blocked, `first_gpu_launch_order` forbids GPU execution, `gpu_vm_sync_manifest`
is blocked, or `gpu_vm_runtime_preflight_plan` is unsafe. Its
`first_gpu_external_input_packet` condenses the remaining live IDs, env vars,
provider secrets, scene artifacts, owner GPU command, and VM checks into one
redacted operator packet and writes `first_gpu_external_input_packet.md` beside
the output manifest when an output path is provided, while its
`first_gpu_operator_actions` mirrors the packet's ordered fix list, and its
`remediation_plan` groups remaining blockers by cross-repo fix lane and names
the evidence or command needed before GPU time is useful.
`blueprint-build-first-gpu-run-packet` now also writes
`gpu_provider_bootstrap.md` and `gpu_provider_bootstrap.json` so the RunPod or
equivalent GPU VM setup, Isaac GPU constraints, and NIM boundary travel with the
owner-command packet. The same packet includes `first_gpu_simulator_path_matrix`
files that distinguish the selected first-GPU backend from Arena/policy,
MuJoCo/PyBullet preflight, Newton, and NIM inference-service roles,
`first_gpu_launch_order` files that prevent running GPU commands before WebApp,
scene, sync, VM-preflight, owner-command, and simulator gates are ready, while
still allowing the owner proof command before post-GPU closure proof exists,
`first_gpu_blocker_resolution` JSON/Markdown files that convert current readiness blockers into an ordered
operator fix list with top-level `actions`, `action_count`, and
`blocked_action_count` fields plus `blocker_details` for hard preflight scene
and GPU-handoff inputs, field-level WebApp upstream ID evidence, and owner
proof wrapper/trace/output requirements, a read-only
`webapp_upstream_truth_verification_commands.sh` script that verifies real
non-placeholder WebApp upstream IDs without mutating artifacts or submitting a
WebApp request, `first_gpu_scene_asset_acquisition` files that name the
World Labs/world-manifest/materialized-asset evidence needed to clear scene
blockers and expose when the source video inputs are ready for a World Labs
request, whether `WORLDLABS_API_KEY` and
`BLUEPRINT_ALLOW_WORLDLABS_PROVIDER_SUBMISSION=true` are configured, and that
the generated provider-submission script remains before GPU spend,
`first_gpu_webapp_handoff` files that pin the upstream-ID,
forwarding-env, optional WebApp forwarding preflight report, staged-request,
and local-rehearsal boundary; the run-packet builder and cross-repo audit can
consume a redacted `ROBOT_EVAL_JOB_REQUEST_FORWARD_PREFLIGHT_REPORT` to prove
URL/token/capture-root configuration evidence without copying secrets into
Pipeline artifacts, and the generated handoff verifier carries that report
instead of requiring the forwarding token in shell output. It
keeps `gpu_spend_decision.gpu_rental_recommended_now=false` when
`local_webapp_rehearsal_only_observed=true`, so a dry-run WebApp request cannot
be mistaken for the real WebApp-forwarded full-E2E gate,
`gpu_vm_runtime_preflight` files that check the GPU VM mount, `nvidia-smi`,
owner command executable, Docker availability, and synced-file hashes before the
owner command runs, and block when the sync manifest is blocked, plus
`gpu_vm_sync_manifest` files that checksum the required raw,
simulation-automation, and run-packet artifacts before a GPU VM handoff.

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

Optional Palatial PhysReady twin request/materialization lane:

```bash
blueprint-build-palatial-physready \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

By default this writes `pipeline/palatial_physready/*` request, cost, lineage,
and validation manifests only. It does not call Palatial or upload captured
images. Live Palatial calls require the explicit double gate:

```bash
BLUEPRINT_ENABLE_PALATIAL_PHYSREADY=true \
PALATIAL_API_KEY=<secret> \
blueprint-build-palatial-physready \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --allow-live-palatial
```

Use `--label microwave --label tote` or `--object-id <object_id>` to focus a
pilot on captured objects that should become PhysReady twins. The Palatial lane
is documented in
[`docs/PALATIAL_PHYSREADY_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/PALATIAL_PHYSREADY_LANE.md).

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
It writes local orchestration manifests only, including an optional
`isaac_lab_arena` Arena Pack review packet. It does not run simulators, download
assets, start training, call providers, or prove robot readiness unless explicit
per-run approvals and dependencies are present.
Agents SDK and Codex SDK paths are gated live-operator surfaces: when SDK,
credential, CLI, and environment gates are present, agents may inspect
manifests/logs, choose deterministic reruns, summarize blockers, route review,
or patch/test code. They still cannot set proof booleans directly.

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

When `blueprint-capture-pipeline --lane current` or a descriptor requesting
`task_evaluation_run` reaches `simulation_automation`, the capture pipeline also
checks `pipeline/robot_eval_job_requests/inbox/` and consumes queued
`robot_eval_job_request.v1` files through the same fail-closed job orchestrator.
No external simulator, live policy, training, upload, or SDK action is performed
unless its explicit env and CLI gates are present.

The job orchestrator reads a robot-team request for policy/container/trace/demo
references, robot profile, task/scenario scope, rights/privacy scope, operation,
simulator preference, training preference, budget, owner system, provenance, and
timestamp alignment. It validates the request, writes a deterministic state
machine under `pipeline/robot_eval_jobs/<job_id>/`, invokes fixture/local
surfaces when allowed, and writes exact blocked manifests for missing evidence
or denied gates. The inbox runner also copies each accepted request under
`pipeline/robot_eval_job_requests/<job_id>/job_request.json` and writes
`pipeline/robot_eval_job_requests/inbox_run_manifest.json`. Fixture provisioner
and fixture simulator paths prove only the repo-local orchestration loop. Vast,
RunPod, GCP, local process, Docker, MuJoCo, PyBullet, Newton, Isaac Sim, Isaac
Lab-Arena, Agents SDK, and Cosmos training paths stay blocked unless their
explicit environment and CLI gates are present.
Live SDK operators log every decision, tool-call summary, command chosen,
refusal, blocker, and proof effect; deterministic accepted artifacts remain the
only source for true proof booleans.

Prepared worker images live under
`deploy/docker/robot_eval_worker/{isaac,mujoco}/`. They run
`blueprint-run-robot-eval-worker`, which loads `BLUEPRINT_EVAL_MANIFEST_URI`,
delegates to the job orchestrator, and copies artifacts before shutdown when an
artifact output URI is provided. Worker manifest input supports local/file,
HTTP(S), GCS, S3, and R2; live RunPod/Vast/GCP workers require a remote
`BLUEPRINT_EVAL_MANIFEST_URI` using `https://`, `gs://`, `s3://`, or `r2://`
because a local path is only a staging artifact. Artifact output supports
local/file, GCS, S3, and R2.
For live/non-fixture provider jobs, the worker fails closed before orchestration
unless `artifact_output_uri` or `--artifact-output-uri` is present, because the
startup contract requires a finalizer destination before GPU time is useful.
Fixture/local workers may opt into the same strict rule with
`artifact_output_uri_required=true` or `--require-artifact-output-uri`.
Live provider workers also require the queued manifest envelope to use
`schema_version: "robot_eval_worker_manifest.v1"` and carry an embedded
`job_request`; a raw job request JSON is not accepted as a provider worker
manifest.
For non-fixture simulators, the manifest also carries a
`runtime_preflight_contract` that must run before scene load and cannot upgrade
proof by itself. Isaac contracts require NVIDIA inventory, driver, Vulkan/RTX,
headless launch, blank-scene load, and test-frame checks; MuJoCo contracts keep
the cheaper path with import/headless/EGL-when-rendering/rollout checks.
`blueprint-run-robot-eval-worker` writes `worker_runtime_preflight.json`; when
simulator execution is explicitly allowed for a non-fixture worker, a missing or
failing runtime preflight command blocks before scene work. The command can be
provided in the worker manifest as `runtime_preflight_command`,
`runtime_preflight_commands.<simulator>`, or through
`BLUEPRINT_RUNTIME_PREFLIGHT_COMMAND`. Preflight stdout/stderr are written as
`worker_runtime_preflight.stdout.log` and
`worker_runtime_preflight.stderr.log`; if preflight blocks before the job
orchestrator runs, the worker still copies those worker-level failure artifacts
to the configured artifact output URI when one is available.
`blueprint-run-robot-eval-job` writes that strict `worker_manifest.json` beside
the provider launch request. For live providers, upload that manifest to object
storage and set `BLUEPRINT_EVAL_MANIFEST_URI` before the provider launcher can
be ready; `gpu_provider_launch_request.json` records both the local staging path
and the fetchable manifest URI plus runtime-preflight contracts.
When the worker runs, it writes `worker_runtime_manifest.json` into the worker
scratch directory, the job directory, and the configured artifact output
destination so the finalizer status travels with the job bundle.
Live provider plans now fail closed unless the selected simulator has a
configured versioned worker image ref, for example
`BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF=registry.example/blueprint/isaac-eval-worker:2026-06-12`
or `BLUEPRINT_MUJOCO_EVAL_WORKER_IMAGE_REF=...`; the generic fallback is
`BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF`. They also require
`BLUEPRINT_EVAL_MANIFEST_URI` for the queued worker manifest and
`BLUEPRINT_ARTIFACT_OUTPUT_URI` for the finalizer destination. A Dockerfile path
alone is build scaffolding, and a local `worker_manifest.json` path alone is not
a provider-launchable input.
Each job writes `gpu_provider_launch_request.json` as a dry-run provider envelope
with worker image, command, env-var names, GPU constraints, timeout, max-worker,
idle-shutdown, and artifact-finalizer requirements. It never stores provider
secret values and does not mean a live GPU provider call happened. These images
and launch requests are startup/runtime scaffolds only; provider-native GPU
evidence remains required for simulator proof.

When `gpu_provider_launch_request.json` reaches `request_manifest_ready`, run a
separate provider launcher instead of teaching the website or job orchestrator
to call RunPod/Vast/GCP directly. The launcher is fail-closed until both
`BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true` and `--allow-provider-launch` are
present, and it only runs the command supplied through
`BLUEPRINT_GPU_PROVIDER_LAUNCH_COMMAND` or `--provider-launch-command`:

```bash
BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH=true \
BLUEPRINT_GPU_PROVIDER_LAUNCH_COMMAND="/path/to/provider-launch-adapter" \
blueprint-run-gpu-provider-launcher \
  --job-dir "$CAPTURE_ROOT/pipeline/robot_eval_jobs/$ROBOT_EVAL_JOB_ID" \
  --allow-provider-launch
```

That command receives non-secret context such as
`BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST`, `BLUEPRINT_EVAL_MANIFEST_URI`,
`BLUEPRINT_ARTIFACT_OUTPUT_URI`, `BLUEPRINT_WORKER_IMAGE_REF`, and the timeout
limits. The launcher writes `gpu_provider_launcher_result.json` plus
`gpu_provider_launcher.stdout.log` and `.stderr.log`, stores no raw command or
secret values, redacts known secret env values from captured stdout/stderr logs,
and does not upgrade simulator, allocation, or robot-readiness proof by itself.
For RunPod, the repo-owned adapter command is
`blueprint-run-runpod-provider-adapter`. It defaults to `--mode dry-run` and
writes `runpod_provider_adapter_result.json` with the serverless `/run` and
GraphQL on-demand Pod request shapes but no API call. Live modes
`--mode serverless-run` and `--mode on-demand-pod` require
`BLUEPRINT_ALLOW_RUNPOD_API_CALLS=true`, `RUNPOD_API_KEY`, and
`--allow-runpod-api-call`; they still only submit/allocate provider work and do
not prove simulator execution, robot readiness, safety, or public claim
upgrades. The adapter also records a `cost_control_policy`: serverless `/run`
payloads can set per-request `executionTimeout`, `ttl`, and `lowPriority`, but
RunPod active workers, max workers, and idle timeout are endpoint-level settings
that must be configured on the endpoint. On-demand Pods do not get provider-native
idle shutdown from the request payload, so the adapter carries the worker env
shutdown controls and requires an external watchdog/owner terminator posture.

Each job also writes `gpu_cost_control_ledger.json` with requested budget,
maximum billable GPU seconds, max workers, timeout, idle-shutdown/watchdog
requirements, concrete idle timeout, concrete external watchdog TTL, estimated
GPU seconds, actual GPU seconds when owner-runtime evidence exists, and the
blockers preventing allocation. A blocked scheduler or missing provider gate
records zero estimated GPU seconds and no live provider calls.

Run `blueprint-audit-robot-eval-startup-architecture --job-dir
<capture-root>/pipeline/robot_eval_jobs/<job_id>` after a job pass to verify the
startup contract in one place. The read-only audit checks the async WebApp queue
boundary, Pipeline scheduler ownership, CPU-preflight gate, worker image/cache
contract, runtime preflight before scene load, provider dry-run envelope,
no-secret policy, timeout/idle-shutdown limits, cost-control ledger, and proof
ceilings without running providers or simulators. `blueprint-run-robot-eval-job`
now writes the same
`startup_architecture_audit.json` into every job directory and surfaces its
status/path in `job_run_manifest.json`; the standalone command remains useful
for re-auditing edited or externally produced job artifacts. When
`worker_runtime_manifest.json` is present after a worker run, the audit also
validates the matching `worker_runtime_preflight.json` schema, status, and
proof-boundary fields.

Each robot-eval job also writes `scenario_eval_matrix.json`. It expands the
requested site/task/scenario scope into concrete scenario-family variation runs
from `simulation_automation/scenario_variation_instances.json`. Robot POV
observations, policy adapter inputs, simulator command environments, live
closure coverage checks, and Post-Training Data Package exports use that matrix
so lighting, object rotation, cart shift, blocked path, human crossing,
forklift, occlusion, glare, missing label, wrong object, and narrow approach
angle cases are not collapsed back into one base scenario.
`policy_execution_manifest.json` and `policy_execution_trace.json` also report
required, covered, and missing `scenario_eval_run_id`s for each selected
robot-team modality and for the aggregate trace. Local reference replays can
prove trace coverage only; live policy proof still requires a gated execution
command/API/container run and accepted owner-system evidence.
When `--arena-results-dir` points at existing Isaac Lab-Arena rollout artifacts,
the job ingests those local results into normalized traces, labels, clips,
metrics, reports, delivery manifests, rerun queues, and a Post-Training Data
Package. That proves package code paths and result ingestion only; simulator
execution, robot policy success, contact/safety validation, and robot readiness
remain false unless separate accepted owner evidence exists.

Real deployment or pilot actuals can be supplied inline on the job request,
through `actual_outcome_manifest_uri` / `deployment_outcome_manifest_uri`, as
`pipeline/robot_eval_inputs/actual_outcome_manifest.json`, or as streamed JSON
files in `pipeline/robot_eval_inputs/deployment_outcomes/inbox/`. The job writes
`deployment_outcome_intake_manifest.json`, `deployment_outcome_ledger.json`,
`sim_vs_real_calibration_report.json`, and
`prediction_vs_actual_deployment_summary.json`, plus a deterministic
`real_world_validation_followup_plan.json` for reruns, missed-failure scenario
updates, robot-team tuning review, and site-modification review. Rerun actions
also produce `real_world_validation_followup_request_queue.json` plus
`robot_eval_job_request.v1` drafts under
`pipeline/robot_eval_job_requests/followup_drafts/<job_id>/`; point
`blueprint-run-robot-eval-job --job-request-inbox` at that draft directory to
process the exact follow-up run/variation requests through the same fail-closed
job runner. The live control plane also scans those follow-up queues and lists a
safe `blueprint-run-robot-eval-job --capture-root ... --job-request-inbox ...`
command in `live_pipeline_external_input_packet.json` and `.md`; it does not
auto-run reruns or upgrade real-world proof. It then reflects the calibration
score on `evaluation_result.json`.
Actual records with a `scenario_eval_run_id` must match a prediction for that
same run before predicted-vs-actual closure can pass; unmatched actual records
are listed as calibration blockers rather than
falling back to same-scenario predictions. Actual records without owner evidence
remain calibration inputs only; live outcome proof requires `evidence_refs`, an
owner proof URI, or an owner/operator attestation on every record.

Every job also writes `live_eval_closure_manifest.json`. This is the
requirement-by-requirement closure audit for the full neutral harness:
site capture, task definitions, scenario library, robot POV generation,
scenario/eval suite, failure labels, standard scorecard methodology, robot-team
policy modalities, simulator engine plugins, WebApp upstream truth, rights and
privacy scope, live simulator execution, live policy execution, real-world
outcomes, predicted-vs-actual calibration, review acceptance, signed delivery,
and safety/contact/physics readiness. The closure remains
`local_artifacts_ready_live_external_blocked` until all live gates have accepted
evidence. Robot POV closure requires coverage of every `scenario_eval_run_id` in
the job matrix, not only a matching observation count. Scenario-library and
scenario/eval-suite closure require each claimed variation row to include
concrete mutation details and engine-adapter mutation operations, or a linked
scenario variation instance that carries them. Failure-label closure
requires every failed attempt or failed `scenario_eval_run_id` in
`normalized_attempt_trace.json` to have a corresponding label in
`failure_labels.json`; an unlabeled failed run remains a package/eval blocker.
Evaluation-methodology closure requires the standard scorecard fields to carry
valid values and shapes: success/calibration scores in `[0, 1]` when present,
non-negative rates/counts/timing samples, and well-formed recovery and
world-model-uncertainty summaries.
Policy-interface closure requires every selected robot-team modality to be
supported, selected, non-blocked, and complete against its modality-specific
reference fields. Live-policy closure additionally requires
`policy_execution_manifest.json` and `policy_execution_trace.json` to agree that
at least one selected modality was actually executed, completed, and proven; a
recorded/reference replay with trace actions is still coverage evidence, not
live policy proof.
Report-generation closure requires `robot_eval_report.json` and `.md` plus
linked core job artifacts whose statuses, counts, scorecard fields, policy
status, real-world outcome status, predicted-vs-actual status, and proof booleans
match the report. A section-complete report stub is not enough.
The simulator-engine plugin gate requires every supported engine in
`simulator_engine_plugin_registry.json` to have a ready adapter contract and
managed execution support; a partial or blocked registry remains a closure
blocker. Predicted-vs-actual closure also blocks when deployment outcome records
carry run-level identifiers that do not match a prediction. Real-world
validation closure recomputes owner evidence and actual-outcome signals from
each ledger row; aggregate `real_world_outcome_proven` booleans alone cannot
upgrade the gate. Live-simulator closure also re-audits owner GPU proof
manifests for required identity/runtime fields, zero exit code, empty
blockers/missing inputs, and all validator-emitted evidence flags; an aggregate
`owner_gpu_simulator_execution_proven` boolean alone cannot upgrade simulator
proof. Signed-delivery closure requires non-placeholder external signed URLs,
storage-upload proof, entitlement verification, and owner/operator attestation.
Rights/privacy closure requires explicit `external_use_allowed=true` plus
owner/operator attestation or a non-placeholder evidence reference; a bare
`accepted=true` or OK status cannot upgrade the gate.
Only a
`live_end_to_end_verified` closure can upgrade
`robot_readiness_proven` or `public_claim_upgrade_allowed` in `proof_boundary.json`.
Owner closure evidence can be supplied inline on the job request, by job-request
URI, directly in the job directory, globally under
`pipeline/robot_eval_inputs/live_eval_closure_evidence.json`, or in the
job-specific staged intake path
`pipeline/robot_eval_inputs/<job_id>/live_eval_closure_evidence.json`.

Standalone closure audit:

```bash
blueprint-audit-live-robot-eval-closure \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture_root>/pipeline/robot_eval_jobs/<job_id>
```

Arena result ingest and package build:

```bash
blueprint-ingest-arena-results \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --arena-results-dir /path/to/isaac-lab-arena-results \
  --scenario-count 500 \
  --shard-size 50
```

Optional OpenAI rollout vision labeling command hook:

```bash
BLUEPRINT_ALLOW_ROLLOUT_VISION_LABELING=true \
blueprint-ingest-arena-results \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --arena-results-dir /path/to/isaac-lab-arena-results \
  --allow-rollout-vision-labeling \
  --vision-labeling-command "blueprint-label-rollout-vision-openai --output-dir ."
```

The OpenAI hook reads `failure_labels.json` and `clips_manifest.json`, extracts
keyframes with `ffmpeg`, calls OpenAI only when `OPENAI_API_KEY` and the rollout
labeling gate are present, and writes `rollout_vision_labels.command.json`.
Ingest consumes those labels as review-required support evidence only.

Optional local delivery command hook:

```bash
BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true \
BLUEPRINT_LOCAL_DELIVERY_ROOT=/var/lib/blueprint/pipeline-control-plane/deliveries \
blueprint-ingest-arena-results \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --arena-results-dir /path/to/isaac-lab-arena-results \
  --allow-delivery-upload \
  --delivery-command "blueprint-deliver-arena-package-local --output-dir ."
```

The local delivery hook copies `delivery_bundle/` to a local delivery root and
returns local access paths. It does not create cloud signed URLs or verify
customer entitlement.

Arena package artifact/proof-boundary audit:

```bash
blueprint-audit-arena-package \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --package-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --expected-scenario-count 500 \
  --require-job-artifacts
```

One-command local fixture smoke:

```bash
blueprint-smoke-arena-package-local --output-dir output/arena-fixture-smoke
```

The smoke creates a synthetic local capture/results fixture, runs the real Arena
ingest CLI path for a 500-scenario schedule, exercises review-required vision
labels, local delivery, fake local operators, and the package audit, then writes
`arena_fixture_smoke_manifest.json`. It proves local package automation only;
it does not prove WebApp upstream truth or owner-system Isaac Lab-Arena
execution.

Live setup and external-gate preflight:

```bash
blueprint-audit-live-pipeline-setup \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --package-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --digitalocean-droplet-name paperclip-prod-01 \
  --digitalocean-droplet-ip 206.81.11.69
```

The setup audit loads local env files without printing secret values, checks
configured commands, owner-supplied Arena result directories, Codex CLI, and SDK availability, and writes
`pipeline/live_pipeline_setup/live_pipeline_setup_manifest.json`. ChatGPT
Pro/Codex OAuth may be used through an authenticated `codex` CLI when
`BLUEPRINT_ALLOW_CODEX_CLI_HOST_OAUTH=true` and the live Codex operator gate are
both set. Repo-local OpenAI SDK calls still require explicit API-key/env
configuration or a command hook that owns its own OAuth flow. The DigitalOcean
droplet can act as an always-on control plane, but it is not GPU/Arena execution
proof by itself.

Use `--arena-results-dir` or `BLUEPRINT_ARENA_RESULTS_DIR` when an owner system
has already produced Isaac Lab-Arena result artifacts. That path can be ready
for result ingest without opening the simulator-execution gate; it still does
not prove simulator execution or robot readiness by itself.

Post-Training Data Package export and archive:

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
