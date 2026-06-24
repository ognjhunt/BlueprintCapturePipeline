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
- `simulation_automation/eval_ready_task_grounding.json`
- `simulation_automation/camera_calibration_quality_gate.json`
- `simulation_automation/robot_fk_projection_manifest.json`
- `simulation_automation/robot_fk_projected_skeleton_trace.jsonl`
- `simulation_automation/handle_proxy_state_check.json`
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
- `robot_eval_jobs/<job_id>/evaluation_substrate_registry.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_evaluation_request.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_provider_runtime_package.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_provider_execution_manifest.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_provider_cost_control_ledger.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_provider_artifact_upload_proof.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_policy_interface_binding.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_rollout_manifest.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_rollout_results.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/vision_success_labels.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_vision_success_review_queue.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_episode_consistency_request.json` when generated
  WAM rollouts are ready for an external forward/inverse consistency scorer
- `robot_eval_jobs/<job_id>/wam_episode_consistency.command.json` when a separate
  episode-consistency scorer command runs
- `robot_eval_jobs/<job_id>/wam_consistency_checks.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/eval_ready_task_grounding.json` when WAM/substrate
  evaluation consumes eval-ready task grounding
- `robot_eval_jobs/<job_id>/camera_calibration_quality_gate.json` when
  WAM/substrate evaluation consumes eval-ready task grounding
- `robot_eval_jobs/<job_id>/robot_fk_projection_manifest.json` when
  WAM/substrate evaluation consumes eval-ready task grounding
- `robot_eval_jobs/<job_id>/robot_fk_projected_skeleton_trace.jsonl` when
  WAM/substrate evaluation consumes eval-ready task grounding
- `robot_eval_jobs/<job_id>/handle_proxy_state_check.json` when WAM/substrate
  evaluation consumes eval-ready task grounding
- `robot_eval_jobs/<job_id>/wam_prediction_outcome_correlation_ledger.json`
  when WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/policy_ranking_scorecard.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_eval_claim_boundary.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/real_world_validation_followup_request.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/srcc_validation_plan.json` when WAM/substrate
  evaluation is requested
- `robot_eval_jobs/<job_id>/wam_real_world_validation_anchor_manifest.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_customer_validation_envelope.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_production_ops_manifest.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/wam_classical_sim_cross_check_plan.json` when
  WAM/substrate evaluation is requested
- `robot_eval_jobs/<job_id>/robot_pov_observation_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_observations.jsonl`
- `robot_eval_jobs/<job_id>/robot_pov_frame_sequence_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_render_storyboard.json`
- `robot_eval_jobs/<job_id>/policy_execution_manifest.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.jsonl`
- `robot_eval_jobs/<job_id>/unitree_policy_stack_installation_audit.json` when
  Unitree policy/provider configuration is probed; this is the aggregate gate
  for whether locomotion, manipulation runtime, and Unitree action-command
  components are all configured
- `robot_eval_jobs/<job_id>/unitree_groot_n17_sonic_installation_audit.json`
  when GR00T N1.7 + UNITREE_G1_SONIC runtime configuration is probed
- `robot_eval_jobs/<job_id>/unitree_groot_n17_sonic_policy_runtime_summary.json`
  and `unitree_groot_n17_sonic_policy_runtime_truth_boundary.json` when the
  GR00T/SONIC policy lane is audited; these are not fresh model execution by
  themselves
- `robot_eval_jobs/<job_id>/unitree_groot_n17_sonic_policy_server_preflight.json`
  when the GR00T/SONIC PolicyServer lane is checked for source, Python
  environment, `UNITREE_G1_SONIC` modality schema, local disk, CUDA, and Sim2Sim
  readiness without downloading weights or launching hardware
- `robot_eval_jobs/<job_id>/policy_action_model_command_output.json` records
  whether the concrete GR00T/SONIC PolicyServer wrapper actually returned an
  action chunk; `gr00t_policy_server_timeout:ping` means the wrapper ran but no
  PolicyServer responded at the configured endpoint
- `robot_eval_jobs/<job_id>/unitree_groot_n17_sonic_sim2sim_execution.json`
  records whether a GR00T/SONIC action chunk was consumed by the simulator-only
  MuJoCo bridge; this is not official GR00T-WholeBodyControl deployment proof
  and not task-success proof by itself
- `robot_eval_jobs/<job_id>/policy_autoresearch/policy_autoresearch_report.json`
- `robot_eval_jobs/<job_id>/policy_autoresearch/agent_idea_tree.json`
- `robot_eval_jobs/<job_id>/policy_autoresearch/policy_candidate_package.json`
- `robot_eval_jobs/<job_id>/policy_autoresearch/heldout_eval_result.json`
- `robot_eval_jobs/<job_id>/policy_autoresearch/followup_real_world_validation_request.json`
- `robot_eval_jobs/<job_id>/policy_improvement_run/policy_improvement_run_offer.json`
- `robot_eval_jobs/<job_id>/policy_improvement_run/policy_improvement_run_offer.md`
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

Run repository commands through the synced environment:

```bash
uv run blueprint-capture-pipeline --help
```

This is a repository development setup only. It is not the supported single-VM GPU runtime bootstrap path.

Optional LLM support for the capture review agent:

```bash
uv sync --extra dev --extra llm
```

Local tests automatically add `src/` and the sibling `BlueprintContracts/src` to `sys.path` through [`tests/conftest.py`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/tests/conftest.py). If the contracts repo is not present beside this repo, install `blueprint-contracts` before running `uv run pytest`.

Cross-repo external alpha gate:

```bash
python scripts/run_external_alpha_launch_gate.py
```

Sim-only beta profile for post-upload autonomy:

```bash
export BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL=true
export BLUEPRINT_SIM_ONLY_BETA_AUTONOMY=true
export BLUEPRINT_ALLOW_SIMULATOR_EXECUTION=true
export BLUEPRINT_MUJOCO_G1_MODEL_ROOT=/path/to/mujoco_menagerie/unitree_g1
```

With this profile, uploads without explicit requested outputs default into `qualification`, `evaluation_prep`, and `simulation_automation`; auto-staged `robot_eval_job_request.v1` work uses the MuJoCo runtime profile; and the control plane can drain accepted WebApp-style job requests into the packaged `blueprint_pipeline.mujoco_g1_simulator_command`. `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION` remains an explicit gate, and a MuJoCo G1 asset root or `BLUEPRINT_MUJOCO_ALLOW_FETCH_G1_ASSETS=true` is required before the packaged command is configured. This proves only sim-only beta execution when the job artifacts contain trace, metric, visual media, and scenario-run coverage evidence. It does not prove physical robot readiness, deployment readiness, or robot-team-grade closure.

Local sim-only beta gate:

```bash
python scripts/run_sim_only_beta_local_gate.py \
  --capture-root /absolute/path/to/capture-root \
  --webapp-repo /Users/nijelhunt_1/workspace/Blueprint-WebApp \
  --mujoco-g1-root /absolute/path/to/mujoco_menagerie/unitree_g1
```

This starts the real local Pipeline intake service with a synthetic token, runs WebApp forwarding preflight with the read-only intake probe, posts a WebApp-built `robot_eval_job_request.v1` through the WebApp route, processes the staged Pipeline inbox, runs the packaged MuJoCo sim-only command, and writes `pipeline/live_pipeline_control_plane/sim_only_beta_local_gate/sim_only_beta_local_gate_report.json`. The report must be `status=passed` before claiming local post-upload autonomy. The report remains local proof only; production forwarding, deployment parity, remote cloud execution, and physical robot readiness require separate evidence.

Sim-only beta release gate:

```bash
python scripts/run_sim_only_beta_deployment_parity_proof.py \
  --capture-root /absolute/path/to/capture-root \
  --route-forwarding-proof /absolute/path/to/production_route_forwarding_proof.json \
  --webapp-url https://<webapp-host> \
  --pipeline-intake-url https://<pipeline-host>/api/live-pipeline/job-requests \
  --webapp-deployed-commit <commit-sha-from-deploy-provider> \
  --pipeline-deployed-commit <commit-sha-from-deploy-provider>

python scripts/run_sim_only_beta_release_gate.py \
  --capture-root /absolute/path/to/capture-root \
  --forwarding-preflight-report /absolute/path/to/forwarding_preflight.json \
  --production-route-forwarding-proof /absolute/path/to/production_route_forwarding_proof.json \
  --production-deployment-proof /absolute/path/to/sim_only_beta_production_deployment_proof.json
```

The deployment/parity proof checks WebApp `/health/ready`, Pipeline intake `/health`, authenticated intake-audit reachability when the intake is routed under `/api/live-pipeline/*`, clean `HEAD == origin/main` repo parity, and deployed commit equality when commit values are supplied. A route-forwarding proof can supply the WebApp URL and forwarding endpoint URL when those fields are present, but deployed commits and the live intake token still come from deployment/runtime configuration. The release gate reads the local sim-only gate report and WebApp forwarding preflight report, then requires a current production route-forwarding proof for the same capture root plus deployment/parity proof before writing `pipeline/live_pipeline_control_plane/sim_only_beta_release_gate_report.json`. The report must be `status=passed` before claiming beta release readiness. Physical robot readiness and remote-cloud provider execution stay out of scope for this sim-only gate.

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

Fixture WAM/substrate evaluation can be requested on the same job path:

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-request /path/to/robot-eval-job-request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate fixture_wam
```

That writes local WAM rollout, fixture vision-label, normalized attempt,
failure-label, policy-ranking scorecard, claim-boundary, and real-world
validation follow-up artifacts. It performs no live provider calls and does not
prove physical robot readiness, deployment approval, safety approval, public
readiness, or customer-specific SRCC.

When `eval_ready_task_grounding.json` is present, the OSCAR/Cosmos WAM evaluator
copies it into the job directory, enriches task prompts with the selected
task-object target, attaches the camera calibration quality gate, consumes the
robot FK/projected-skeleton trace as action-conditioning support, records the
lightweight articulated-object proxy check, and writes
`wam_prediction_outcome_correlation_ledger.json`. These files ground and audit
the learned rollout, but they stay support artifacts: calibration gates,
projected skeletons, VLM labels, and handle proxies do not prove physical
contact, torque, safety, deployment approval, or real-world task success.

The MuJoCo Unitree policy/WAM closed-loop helper now has a default local
OSCAR-style support backend for the no-live-provider case. If no gated
OSCAR/Cosmos WAM command is configured, `run_robot_policy_wam_closed_loop_attempt`
generates action-conditioned next-observation frames plus short MP4 segments,
records the policy action, simulated proprioception keys, and projected G1
skeleton support in `wam_generated_next_observations.jsonl`, then re-queries
the selected Unitree policy on those generated frames. Those artifacts are labeled
`default_local_wam_generator_used=true` and
`learned_oscar_or_cosmos_model_ran=false`; they are useful loop evidence, not a
claim that a learned OSCAR/Cosmos checkpoint or physical robot sensor loop ran.

Forward/inverse episode consistency is a separate scorer layer, not a property
claimed by WAM execution or by the evaluator itself. The OSCAR/Cosmos WAM
evaluator writes `wam_episode_consistency_request.json`; a separate VLM or human
review command writes `wam_episode_consistency.command.json`; the evaluator then
normalizes that result into `wam_consistency_checks.json`. See
[`docs/WAM_EPISODE_CONSISTENCY_SCORER.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/WAM_EPISODE_CONSISTENCY_SCORER.md).

For Unitree G1, the robot-policy lane is Unitree-native. Use `unitree_g1_policy`
for locomotion/control, `unitree_groot_n17_sonic_policy` as the current top
GR00T N1.7 + UNITREE_G1_SONIC manipulation/action-command candidate,
`unitree_lerobot_policy` for G1 Dex1/Dex3/gripper manipulation, and
`unitree_unifolm_vla_policy` or `unitree_unifolm_wma_policy` only when a
Unitree command/checkpoint is available. UnifoLM VLA readiness requires a VLA
checkpoint via `BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT` or the provider-facing
alias `BLUEPRINT_UNITREE_UNIFOLM_POLICY_CHECKPOINT`, plus the VLM backbone at
`BLUEPRINT_UNITREE_UNIFOLM_VLM_CHECKPOINT`.

GR00T/SONIC readiness requires `BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT`,
`BLUEPRINT_UNITREE_GROOT_N17_SONIC_WBC_ROOT`,
`BLUEPRINT_UNITREE_GROOT_N17_CHECKPOINT`,
`BLUEPRINT_UNITREE_G1_SONIC_CHECKPOINT`, and a runnable
`BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND`, usually
`blueprint-unitree-groot-n17-sonic-policy-server-command`. Manipulation proof
also requires a reachable GR00T PolicyServer and a configured SONIC Sim2Sim
command. Use `blueprint-run-unitree-groot-n17-sonic-policy-server-preflight` to
record whether the source checkout, server venv, `UNITREE_G1_SONIC` schema,
disk, CUDA, and Sim2Sim prerequisites are ready. A preflight pass is still not
model execution; do not replace the proven Unitree RL Gym locomotion path with
GR00T/SONIC until a separate simulator proof actually runs the GR00T/SONIC
command and emits action chunks. The current third-party GR00T/SONIC proof can
emit 3120-value action chunks, consume them through a simulator-only MuJoCo
bridge, and re-query the policy through a WAM loop; it still does not prove an
official/trusted checkpoint, physical readiness, deployment approval, safety
validation, or correctly placed-object task success.

Do not use OpenVLA, OSCAR, Cosmos, fixture WAM, or generated WAM rollouts as the
G1 robot policy. OpenVLA can remain a generic comparison candidate for
non-Unitree work, and WAM outputs remain evaluator/support artifacts unless a
separate Unitree-specific policy endpoint consumes the observation and emits
normalized G1 actions. For this machine, source `.env.unitree.local` to bind the
verified local Unitree RL Gym root and checkpoint before running G1 MuJoCo policy
proofs. See
[`docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md`](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md).

Use the explicit provider-registry fields instead of the legacy
`selected_provider` field when answering what controls the G1:
`selected_locomotion_provider`, `selected_unitree_manipulation_runtime`,
`selected_unitree_action_command`, `selected_unitree_hand_policy`,
`unitree_hand_manipulation_policy_in_place`, `openvla_selected_for_g1_policy`,
`wam_selected_for_g1_policy`, and `g1_robot_policy_family_decision`.

Current Unitree manipulation status: the repo has Unitree-specific adapter,
provider-bundle, SDPA fallback image plumbing, and a live Unitree UnifoLM
`/act` server bridge path. A RunPod `provider_bundle_kind=unitree_unifolm`
smoke completed and returned a Unitree UnifoLM `manipulation_contact` action
with a 25-step, 23-value action chunk at
`robot_eval_jobs/unitree_unifolm_provider_import_20260622T173653Z_framefix_default_retry/`.
That proves one Unitree model-backed action-command execution.

The latest strict endpoint/WAM requery artifacts keep fresh inference separate
from provider replay. The MuJoCo smoke under
`robot_eval_jobs/mujoco_g1_unitree_unifolm_live_endpoint_1ep_every_step_20260622T210403Z/`
uses the authenticated endpoint path with `endpoint_policy_used=true`,
`fixture_policy_used=false`, `endpoint_invocation_count=4`, and zero rejected
actions. Its manipulation report records
`unitree_endpoint_hand_policy_output_observed=true`,
`unitree_endpoint_hand_policy_used=false`,
`unitree_endpoint_fresh_policy_action_command_ran=false`, and
`unitree_endpoint_provider_output_replay_used=true`.

That means Unitree-family action output and action chunks are flowing through
the endpoint, but a fresh per-observation Unitree hand/manipulation policy is
not yet proven in that run. `unitree_endpoint_hand_policy_used=true` is reserved
for a live Unitree-specific command, server, or provider call that runs for the
current observation, not for an imported provider result. The WAM requery proof
under
`robot_eval_jobs/oscar_wam_unitree_unifolm_every_step_requery_strict_20260622T211219Z/`
therefore remains blocked with
`blocked_policy_requery_provider_replay_not_fresh_unitree_hand_policy` even
though an endpoint action was returned for the WAM-generated first frame.

The intended fresh endpoint path is still Unitree-native: launch a long-lived
UnifoLM `/act` server with `blueprint-launch-unitree-unifolm-runpod-server
launch`, then call it from the local endpoint through
`blueprint-unitree-unifolm-vla-server-bridge --server-url
https://<pod_id>-8777.proxy.runpod.net/act`, or use an equivalent Unitree
LeRobot/GR00T-SONIC command endpoint. OpenVLA remains only a comparison
candidate, and OSCAR/Cosmos/WAM outputs remain evaluator artifacts, not the G1
robot policy.

The fixture WAM evaluator can also be run directly against an existing job
directory:

```bash
blueprint-run-wam-fixture-evaluator \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --evaluation-substrate fixture_wam
```

Live or owner-provided WAM adapters use the same job path but remain gated. For
example, a Cosmos-style adapter must be enabled explicitly and supplied through
env/CLI; both `--allow-wam-provider` and
`BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true` are required. Provider credentials stay
in env and are never written to artifacts:

```bash
BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true \
BLUEPRINT_COSMOS3_WAM_API_KEY=<redacted> \
blueprint-run-robot-eval-job \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-request /path/to/robot-eval-job-request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate cosmos3_wam \
  --allow-wam-provider \
  --wam-provider-command cosmos3_wam="/path/to/cosmos_wam_adapter" \
  --wam-artifact-output-uri gs://customer-bucket/<job_id>/wam
```

The adapter receives `BLUEPRINT_WAM_PROVIDER_INPUT` and
`BLUEPRINT_WAM_PROVIDER_OUTPUT` and must write provider rollout JSON. Missing
adapter commands, missing auth envs, failed commands, timeouts, invalid JSON, or
empty rollout outputs write blocked WAM provider manifests instead of making a
readiness claim.

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
Each job also writes `gpu_startup_pipeline_plan.json`. This is the managed-GPU
startup policy for website-origin jobs: the WebApp queues and forwards only,
BlueprintCapturePipeline owns provider selection and spend gates, and customer
scene load waits for runtime preflight/canary evidence. The default provider
posture is managed capacity first: RunPod Secure Cloud or a pinned dedicated
RunPod endpoint for near-term bursts, Lambda Cloud as a second managed lane,
AWS G6/G5-class instances when hyperscaler controls matter, and CoreWeave for
reserved/scale AI infrastructure. Vast and community marketplace capacity are
not the default customer path; they remain experiment or overflow lanes and
must fail closed unless an explicit strict-preflight/canary override is present.
The startup plan now includes `provider_worker_session_policy`: allocate a
provider worker once per evaluation job or worker role, wait for `/readyz`, send
all repeated policy calls to `/infer`, then request `/shutdown` after final
artifacts are uploaded. `blueprint-build-provider-worker-contract` can write the
standalone `provider_worker_contract.json`, and
`blueprint-build-provider-worker-endpoint-manifest` can write the provider
endpoint/discovery side artifact that RunPod and Vast adapters also emit as
`provider_worker_endpoint_manifest.json`. That manifest is endpoint discovery,
not allocation, readiness, teardown, cost, safety, or deployment proof.
`blueprint-provider-worker-policy-command-adapter` lets existing policy-command
loops call an already-running worker via `BLUEPRINT_PROVIDER_POLICY_WORKER_URL`
and `BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL`. For explicit batch checks,
`blueprint-run-provider-worker-session` waits for `/readyz` once, sends multiple
`/infer` requests, and optionally asks `/shutdown` once while still requiring a
provider teardown artifact for cost proof. One-shot provider launchers, such as
a Vast command that rents an instance for a single policy action, are blocked
from repeated WAM/policy loops rather than silently cold-starting per inference.
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
contract, managed GPU startup policy, marketplace fail-closed posture, runtime
preflight before scene load, provider dry-run envelope, no-secret policy,
timeout/idle-shutdown limits, cost-control ledger, and proof ceilings without
running providers or simulators. `blueprint-run-robot-eval-job` now writes the same
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

Sim-only policy autoresearch can run after `scenario_eval_matrix.json` exists:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --reviewed-examples /path/to/reviewed_success_failure_examples.json \
  --simulator-engine mujoco \
  --simulator-engine isaac_sim \
  --evaluator-command "python /path/to/site_policy_eval_runner.py" \
  --max-iterations 8 \
  --agent-count 4
```

Minimal seed recipe:

```json
{
  "schema_version": "policy_autoresearch_recipe.v1",
  "policy_id": "site_policy_seed",
  "policy_kind": "code_as_policy_navigation_heuristic",
  "mutable_parameters": {
    "planner": "direct",
    "clearance_margin_m": 0.05,
    "dynamic_obstacle_yield": false,
    "perception_vote_count": 1,
    "retry_budget": 0,
    "max_speed_mps": 0.9,
    "grasp_alignment_correction": false
  }
}
```

This lane freezes a verifier from the scenario matrix, splits train and heldout
runs, optionally freezes reviewed success/failure examples into that verifier,
mutates only policy recipe parameters, and promotes a candidate only when heldout
task success reaches the configured target and safety/contact gates stay clean.
Candidate recipes that include reward/verifier/classifier override keys are
blocked before the loop runs. It writes `policy_autoresearch_report.json`,
`agent_idea_tree.json`, `policy_candidate_package.json`, `heldout_eval_result.json`, and
`followup_real_world_validation_request.json` under
`policy_autoresearch/`. The candidate package is a Task Evaluation Run support
artifact only: it does not upgrade simulator execution, live policy execution,
physical robot readiness, safety validation, or public claims without separate
accepted owner-system evidence.

When `--evaluator-command` is supplied, the command is called for each candidate
and split with `BLUEPRINT_POLICY_AUTORESEARCH_RECIPE`,
`BLUEPRINT_POLICY_AUTORESEARCH_MATRIX`, `BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT`,
`BLUEPRINT_POLICY_AUTORESEARCH_PHASE`,
`BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE`, and
`BLUEPRINT_POLICY_AUTORESEARCH_VERIFIER_SHA256`,
`BLUEPRINT_POLICY_AUTORESEARCH_CAPTURE_ROOT`,
`BLUEPRINT_POLICY_AUTORESEARCH_JOB_DIR`, and
`BLUEPRINT_POLICY_AUTORESEARCH_SOURCE_MATRIX` in its environment. If
`--evaluator-attempt-trace` is supplied, it is also exposed as
`BLUEPRINT_POLICY_AUTORESEARCH_ATTEMPT_TRACE`. The evaluator must write JSON
with `attempts`, `results`, or `episodes` so the lane can normalize task
success, failure modes, safety events, and contact events against the frozen
verifier.

For cheap local smoke tests, the packaged replay evaluator can consume existing
attempt evidence without claiming fresh simulator execution:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_local_evaluator" \
  --evaluator-attempt-trace /path/to/simulator_command_batch_attempt_trace.jsonl \
  --max-iterations 2 \
  --agent-count 2
```

Without this hook, the lane uses the built-in deterministic recipe evaluator
for local contract tests and dry runs.

Policy Improvement Runs package the commercial offer that sits one step above
the baseline Task Evaluation Run and Post-Training Data Package. A robot team
supplies its policy or base model, robot embodiment, action interface, target
task, success threshold, and cycle-time threshold. Blueprint evaluates the
baseline, diagnoses dominant failures, creates twin/cousin scenarios and a
curriculum, post-trains or lifts a bounded candidate, tests that candidate on
heldout/sealed scenarios, and emits an improved artifact plus evidence report.
The contract stays model-agnostic and customer-supplied-policy friendly:
`black_box` accepts an API/container/action-trace surface, `config_adapter`
accepts adapter or task-head access, and `source_training` is the only mode that
requires source/training access.

Private hardware integration is also explicit. For closed robots, the preferred
default is `customer_hosted_sealed_eval_capsule`: the robot team keeps its
URDF/MJCF/USD, controller, simulator, and hardware bridge private while
Blueprint sends a least-privilege task/scenario/eval packet and receives
normalized owner proof. Blueprint does not export raw capture bundles, full
scene assets, full scoring harnesses, or sealed audit seeds by default. If the
customer shares a private Robot Embodiment Pack under NDA, use
`private_asset_hosted_by_blueprint`; if the customer runs a physical robot, use
`physical_robot_evidence_bridge` and require camera/action/outcome evidence
joined to exact `scenario_eval_run_id` values.

```bash
blueprint-build-policy-improvement-run \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --access-level config_adapter \
  --customer-policy-ref customer-tote-policy-v3 \
  --embodiment g1-humanoid \
  --action-interface joint_position_delta_20hz \
  --hardware-integration-mode customer_hosted_sealed_eval_capsule \
  --site-ip-protection-level sealed_eval_capsule \
  --customer-hosted-connector-ref gs://robot-team/blueprint/connector-contract.json \
  --target-task tote-transfer \
  --success-threshold 0.95 \
  --cycle-time-threshold-seconds 90 \
  --improvement-target adapter \
  --improvement-target task_head
```

The builder writes `policy_improvement_run_offer.json`,
`private_hardware_integration_plan.json`, `policy_improvement_run_offer.md`, and
`policy_improvement_run_webapp_summary.json` under `policy_improvement_run/`.
The manifest binds together the scenario matrix, normalized baseline attempts,
standard evaluation scorecard projection, failure labels, Post-Training Data
Package export, policy-autoresearch candidate package, heldout result, staged
readiness ladder, WebApp-safe summary projection, and proof boundary. It can say
the run is ready for baseline evaluation, failure diagnosis, post-training
package build, policy-autoresearch, candidate promotion, or customer review. It
cannot turn sim heldout success into deployment approval: sealed audit scenarios
must remain outside training, and robot readiness, physical safety validation,
real-world outcome, and public claim upgrades remain false until separately
proven by accepted live evidence.

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
