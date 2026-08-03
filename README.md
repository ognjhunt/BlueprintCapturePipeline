# BlueprintCapturePipeline

`BlueprintCapturePipeline` turns truthful raw captures into maintained Site-Task
Testbeds, then routes claim-level **Task Evaluation Runs** to qualified evidence
methods. Every run returns a decision, a partial decision, or an explicit
abstention. Rights-cleared evidence may be exported for evaluation or
post-training use inside the run; that export is not another product and does
not imply training or policy improvement.

Two lanes are active today.

The site/package lane is: `BlueprintCapture` output -> privacy-safe World Labs input prep -> World Labs API upload/request -> persisted provider manifests -> materialized World Labs output assets with checksums -> CPU/pre-GPU scene and episode preflight -> simulation automation manifest -> explicitly gated simulator runs.

The decision/evidence lane is: a provider-neutral request bound to an exact
testbed -> deterministic claim decomposition and qualification -> the cheapest
sufficient combination of geometry, captured observations, traditional
simulation, learned/world-model evaluation, provider tools, or accepted physical
evidence -> normalized results -> a Decision Envelope. Cosmos, OSCAR, MuJoCo,
Isaac, and future methods are candidates behind replaceable profiles; a default
or runnable backend is not evidence that it is qualified.

Older scene-memory, retrieval/alignment, Cosmos-Predict2.5 (`cosmos_wam`), single-VM GPU, SimReady, and Marble bridge lanes are legacy/advisory support paths unless a command or artifact explicitly requests them.

For public language, Google/Meta smart glasses are supported only for approved repeat walkthroughs where the assignment, hardware, launch proof, and downstream capture/package proof exist. This repo treats glasses outputs as partial/internal until that proof chain exists.

Doctrine and strategy (read in this order; [`docs/DOCTRINE_PRECEDENCE.md`](docs/DOCTRINE_PRECEDENCE.md) governs conflicts):

- [`PLATFORM_CONTEXT.md`](PLATFORM_CONTEXT.md) — what is true and sellable today
- [`WORLD_MODEL_STRATEGY_CONTEXT.md`](WORLD_MODEL_STRATEGY_CONTEXT.md) — model-backend posture and build priorities
- [`VISION.md`](VISION.md) — the long-horizon ladder (rungs 1–5); direction and bets, never overrides the two above
- [`AGENTS.md`](AGENTS.md) — canonical working rules for agents and human engineers
- [`AUTONOMOUS_ORG.md`](AUTONOMOUS_ORG.md) — org roles and agent lanes

AI and engineer orientation maps live under [`docs/architecture/`](docs/architecture):

- [`ai-onboarding-map.md`](docs/architecture/ai-onboarding-map.md)
- [`source-of-truth-map.md`](docs/architecture/source-of-truth-map.md)
- [`command-safety-matrix.md`](docs/architecture/command-safety-matrix.md)
- [`refactor-hotspots.md`](docs/architecture/refactor-hotspots.md)
- [`evaluation-run-interface.md`](docs/architecture/evaluation-run-interface.md)
- [`decision-evidence-router.md`](docs/architecture/decision-evidence-router.md)
- [`task-site-measurement-routing.md`](docs/architecture/task-site-measurement-routing.md)
- [`measurement research monitoring runbook`](docs/runbooks/measurement-research-monitor.md)
- [`measurement adapter execution runbook`](docs/runbooks/measurement-adapter-execution.md)
- [`task-evaluation-supervisor.md`](docs/architecture/task-evaluation-supervisor.md)
- [`TASK_EVALUATION_SUPERVISOR_RUNBOOK.md`](docs/TASK_EVALUATION_SUPERVISOR_RUNBOOK.md)

The operational iPhone/LiDAR reconstruction path—canonical Raw Contract V3.2
or explicitly bounded ARKitScenes proxy input, deterministic cross-platform
worker transport, allocator-bound Postshot/Splatfacto training, exact-camera
evaluation, and registered-appearance production—is documented in
[`docs/CANONICAL_V32_TO_3DGS.md`](docs/CANONICAL_V32_TO_3DGS.md).

Provider-neutral Task Evaluation Run control-plane commands:

```bash
blueprint-route-task-evaluation plan --request request.json --testbed testbed.json \
  --method-profile method.json --qualification qualification.json --output-dir out/plan
blueprint-route-task-evaluation execute --plan out/plan/evidence_plan.json \
  --request request.json --testbed testbed.json --method-profile method.json \
  --qualification qualification.json --fixture-adapter-registry fixture-adapters.json \
  --allow-fixture-adapters --output-dir out/evidence
blueprint-route-task-evaluation aggregate --request request.json --testbed testbed.json \
  --plan out/plan/evidence_plan.json --result out/evidence/result-step.json \
  --output-dir out/decision
blueprint-route-task-evaluation supervise --capture-build /path/to/completed-capture \
  --mode shadow --output-dir out/supervisor
```

`execute` is deliberately hermetic in v1: it requires explicit fixture-adapter
authorization, performs no provider discovery, paid compute, or physical run,
and fails closed otherwise.

The Teleport reconstruction API is available only as a candidate-only external
appearance lane through the canonical paid allocator. It uploads an immutable
RGB-only public-data ZIP, retrieves provider-native PLY and COLMAP camera
metadata, aligns from candidate camera correspondences, evaluates sealed
held-out views only after retrieval, and always attempts deletion. A Teleport
`READY` state does not prove metric scale, collision geometry, Isaac
compatibility, task success, physical truth, or deployment readiness. See the
[`Teleport provider reconstruction runbook`](docs/runbooks/teleport-provider-reconstruction.md).

`supervise` uses OpenAI Agents SDK as the required harness for the durable
manager and all six registered specialist agents. The manager observes each
validated specialist result and selects only the next specialist whose
deterministic prerequisites are present. A capture build alone starts with
claim interpretation and capture/testbed inspection; absent task, robot,
success, rights, or evidence contracts then produce a typed clarification or
blocker instead of invoking irrelevant specialists. Live SDK inference additionally requires
`--allow-live-agent-sdk` and the shared live-operator environment gate. Agent
inference also requires a positive `--agent-inference-budget-usd`; each call
persists a conservative worst-case reservation before provider execution.
Interrupted calls retain that reservation and cannot be silently repeated after
restart. Agent output is advisory and cannot alter the deterministic proof
result.

The normal `run_e2e` capture-build path always enters this supervisor lifecycle
after capture processing and records its status and artifacts in the stage
ledger. There is no alternate production harness or flag that skips the
supervisor. The lifecycle uses `execute_non_spend`, so a live, budget-authorized
SDK run can materialize safe clarification, targeted-recapture, scenario, and
local compilation artifacts immediately from the capture build. Missing live-
inference authority is recorded as a typed blocker; it never silently falls back
to a different harness or deterministic pseudo-agent. This is lifecycle v3 with
a versioned run identifier bound to the capture digest and exact execution
profile; existing v1 and v2 artifacts remain immutable and are not resumed
under broader authority.

`advise` runs the same SDK manager and eligible specialists, but exposes no
callable tools and executes no action. Blueprint validates each proposed action
against the registered tool contract; a valid proposal is recorded as
`requires_operator_approval`, while an unregistered, malformed, oversized, or
proof-changing proposal is refused. Approval is not implicit: an operator must
issue the appropriate validated receipt and start a separately authorized
execution mode.

`execute_non_spend` exposes only capability-scoped registered tools through the
Agents SDK. These include proof-safe reads plus deterministic materialization of
a bound Evidence Plan and its validated leaf Evaluation Run specs into the
supervisor's own generated-artifact directory, plus a review-only targeted
recapture proposal that cannot start capture or infer rights. Every call
produces a digest-bound, zero-cost typed observation with `proof_effect=none`;
no shell, arbitrary filesystem, network, provider, paid, physical-action, or
proof-mutation tool is available.

`execute_preauthorized` additionally requires an operator-issued authorization
receipt and an injected provider-neutral recovery controller. Recovery is bound
to an immutable commit and input digests and is limited by receipt-bound
provider/action allowlists, spend, controller-clock expiry/TTL, retries,
watchdog, and mandatory teardown with explicit provider-zero proof. Vast is the
preferred first live canary backend based on prior Blueprint execution evidence;
RunPod remains a separately qualified fallback. No provider is chosen merely
because its adapter exists. The generic allocator's older RunPod default applies
only to its RunPod-specific strict-smoke launcher; it is not a supervisor
default. The concrete supervisor Vast adapter reuses the authorized Vast WAM
runner and requires versioned result/teardown artifacts plus terminal watchdog
closure; it does not introduce a parallel provider launcher.

An operator receipt can be returned with `supervise --authorization-request
... --authorization-receipt ...`; this only records and replays the strict,
digest-bound response. It never grants SDK tool authority or constructs the
controller. Any injected recovery controller must carry the identical receipt,
and remains responsible for expiry, spend, provider/action, retry, watchdog,
and teardown enforcement.

Agentic robot stacks enter evaluation through
`blueprint_agentic_candidate_policy@1`. Direct policy, decomposed planner+policy,
and verify/recover supervisor candidates are frozen and compiled against the
same scenario manifest, evaluator, predicates, claim ceiling, and hidden-test
separation. Candidate runtime configuration is separately digest-bound and all
operator/paid-resource admission validates before execution artifacts are
created. Candidate agents receive no evaluator or proof authority. The external
Pigey adapter pins a clean exact checkout, excludes the candidate's own success
field, and marks candidate-reported usage as non-authoritative. The concrete
OpenAI project/API-key cost authority writes a digest-bound maximum reservation
before candidate execution, takes a provider-reported zero-cost baseline, and
requires delayed independent reconciliation after the reporting window;
candidate-reported usage cannot settle it. Live use remains blocked without a
dedicated cost scope, rights-holder permission, an operator receipt, and
explicit paid-execution authorization.

The model-neutral, fail-closed composition contract for scientific sim ranking,
provider execution, buyer delivery, teardown, and billing is documented in
[`docs/EVALUATOR_QUALIFICATION_WORKFLOW.md`](docs/EVALUATOR_QUALIFICATION_WORKFLOW.md).
It keeps OSCAR, SC3, Cosmos, future evaluator backends, and compute providers
replaceable and never inherits paper metrics as Blueprint results.

The model-neutral RoboWorld-inspired progress rubric, criterion-scoped view
authority, segment-aggregation ablations, blinded judge-calibration campaign,
hierarchical uncertainty report, and frozen future-backend admission boundary
are documented in
[`docs/ROBOWORLD_EVALUATOR_INTEGRATION.md`](docs/ROBOWORLD_EVALUATOR_INTEGRATION.md).
This evaluator work does not implement Step Forcing or inherit RoboWorld's
published rank-correlation results.

Robot-team buyers: what a Task Evaluation Run and its optional evidence-use
exports contain, how to verify them, and their claim boundaries are documented in
[`docs/BUYER_PACKAGE_TRUST_GUIDE_2026-07-04.md`](docs/BUYER_PACKAGE_TRUST_GUIDE_2026-07-04.md).
Every package export writes a fail-closed `buyer_package_readout.json` +
`buyer_package_summary.md` and `replay_review_instructions.md`.

Paid GPU operators can configure first-class GCP Compute Engine and AWS EC2
adapters using [`docs/GCP_AWS_GPU_PROVIDER_SETUP.md`](docs/GCP_AWS_GPU_PROVIDER_SETUP.md).
The current customer-facing lane uses RunPod Secure active workers, an exact
cached worker digest, same-session readiness evidence, and an atomically leased
warm pool; see
[`docs/runbooks/production-gpu-startup-and-warm-pool.md`](docs/runbooks/production-gpu-startup-and-warm-pool.md).
The durable image layout separates a cached Isaac/robot foundation, an external
checksum-verified model volume, and a thin Blueprint release; see
[`docs/runbooks/groot-oscar-thin-release.md`](docs/runbooks/groot-oscar-thin-release.md).
The production reliability golden path, campaign state machine, artifact
contract, SLOs, ownership, and promotion rules are defined in
[`docs/PRODUCTION_GPU_RELIABILITY_OPERATING_MODEL.md`](docs/PRODUCTION_GPU_RELIABILITY_OPERATING_MODEL.md).

## Scope

Primary product path:

- raw capture materialization from `BlueprintCapture`
- required OpenAI Agents SDK Task Evaluation Supervisor ingress for every
  completed capture build
- Gemini-backed multimodal capture review
- capture evidence analysis and agent review
- deterministic QA aggregation and trust/provenance assembly
- robot-evaluation/evidence-use fit scoring and capturer payout recommendation
- optional provider preview routing
- privacy-safe World Labs input preparation
- World Labs upload/request/operation/world manifest persistence
- World Labs output asset materialization into local checksum/provenance manifests
- webapp sync for buyer-review surfaces
- Site Cards, Task Cards, Scenario Cards, Eval Cards, rights packets, and proof boundaries
- optional rights-gated evidence-use exports such as curated clip/label support
- Decision/Evidence Requests, Evidence Plans, normalized results, Decision
  Envelopes, and append-only physical-outcome joins
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
- `scene_wam_policy_episode_packet/scene_wam_policy_episode_packet.json`
- `scene_wam_policy_episode_packet/initial_policy_observation.json`
- `scene_wam_policy_episode_packet/initial_policy_observation_render.json`
- `scene_wam_policy_episode_packet/scene_episode_task_manifest.json`
- `scene_wam_policy_episode_packet/scene_policy_wam_claim_boundary.json`
- `scene_wam_policy_episode_packet/capture_derived_robot_pov_synthesis/<task_robot_profile>/capture_derived_robot_pov_synthesis_manifest.json`
- `scene_wam_policy_episode_packet/capture_derived_robot_pov_synthesis/<task_robot_profile>/capture_derived_robot_pov_quality_report.json`
- `scene_wam_policy_episode_packet/capture_derived_robot_pov_synthesis/<task_robot_profile>/capture_derived_robot_pov_contact_sheet.jpg`
- `scene_wam_policy_episode_packet/capture_derived_robot_pov_synthesis/<task_robot_profile>/capture_derived_robot_pov_source_qa.json`
- `scene_wam_policy_episode_packet/capture_derived_robot_pov_synthesis/<task_robot_profile>/capture_derived_robot_pov_recapture_guidance.json`
- `robot_eval_jobs/<job_id>/job_request.json`
- `robot_eval_jobs/<job_id>/evaluation_run_spec.json`
- `robot_eval_jobs/<job_id>/evaluation_run_plan.json`
- `evaluation_run_execution.json` when the canonical Evaluation Run execution authority is used
- `evaluation_runs/<run_id>/evaluation_run_{spec,plan,execution}.json` for legacy robot-eval inputs translated through the canonical authority
- `robot_eval_jobs/<job_id>/job_validation.json`
- `robot_eval_jobs/<job_id>/job_plan.json`
- `robot_eval_jobs/<job_id>/agent_orchestration_plan.json`
- `robot_eval_jobs/<job_id>/scheduler_decision.json`
- `robot_eval_jobs/<job_id>/worker_launch_plan.json`
- `robot_eval_jobs/<job_id>/worker_manifest.json`
- `robot_eval_jobs/<job_id>/gpu_provisioning_request.json`
- `robot_eval_jobs/<job_id>/gpu_provider_launch_request.json`
- `robot_eval_jobs/<job_id>/gpu_provider_launcher_result.json` for historical
  legacy-launcher records; the public paid launcher is now hard-disabled
- `robot_eval_jobs/<job_id>/runpod_provider_adapter_result.json` for adapter
  dry-run/read-only records or canonical allocator-owned execution
- `robot_eval_jobs/<job_id>/gpu_cost_control_ledger.json`
- `robot_eval_jobs/<job_id>/provider_closure_audit_report.json` when
  `blueprint-audit-provider-closure` is run
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
- `persistent_wam_short_visual_sanity_manifest.json` when the short
  review-quality WAM visual sanity command is run before longer learned-WAM
  rollouts
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
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_derived_observation_bundle.json`
  when the WAM-derived perception/observation harness runs
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_derived_observation_manifest.json`
  when the WAM-derived perception/observation harness runs
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_perception_harness_checks.json`
  when the WAM-derived perception/observation harness runs
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_policy_observation_adapter_report.json`
  when the WAM-derived perception/observation harness runs
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_derived_observation_steps.jsonl`
  when the WAM-derived perception/observation harness runs
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_perception_backend_request.json`
  when an optional external perception harness backend is explicitly enabled
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_perception_backend_result.json`
  when an optional external perception harness backend is explicitly enabled
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_perception_harness_validation_report.json`
  when the WAM-derived harness writes validation metrics or records that
  validation labels were not supplied
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_false_success_reduction_metrics.json`
  when the WAM-derived harness compares plain generated-video false-success
  labels against harness-gated scoring on supplied validation rows
- `robot_eval_jobs/<job_id>/robot_policy_wam_closed_loop/wam_derived_observation_harness/wam_perception_harness_review_report.md`
  when the WAM-derived harness writes the reader-facing reliability, adapter,
  validation, and claim-boundary report
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
- `robot_eval_jobs/<job_id>/robot_camera_profile_registry.json`
- `robot_eval_jobs/<job_id>/robot_camera_profile_launch_readiness.json`
- `robot_eval_jobs/<job_id>/robot_pov_observations.jsonl`
- `robot_eval_jobs/<job_id>/robot_pov_frame_sequence_manifest.json`
- `robot_eval_jobs/<job_id>/robot_pov_render_storyboard.json`
- `robot_eval_jobs/<job_id>/policy_execution_manifest.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.json`
- `robot_eval_jobs/<job_id>/policy_execution_trace.jsonl`
- `robot_eval_jobs/<job_id>/sc3_eval_protocol.json` records the SC3-style
  evaluator protocol contract, required data, correlation/anchor gates, and
  robot/policy adapter boundaries without launching a model or upgrading
  generated support evidence
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
- `robot_eval_jobs/<job_id>/rl_post_training_handoff_packet.json`
  when the deprecated internal candidate-generation path builds a Task
  Evaluation Run handoff packet
- `robot_eval_jobs/<job_id>/policy_improvement_run/policy_improvement_run_offer.json`
- `robot_eval_jobs/<job_id>/policy_improvement_run/policy_improvement_run_offer.md`
- `robot_eval_jobs/<job_id>/policy_improvement_run/rl_post_training_handoff_packet.json`
  with sparse reward, concurrent baseline A/B, bottleneck, speed curriculum,
  action-chunk QA, and intervention/safety support signals
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
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json`
  when visual augmentation support is prepared
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_variant_requests.jsonl`
  when visual augmentation support is prepared
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/model_backend_registry.json`
  when visual augmentation support is prepared
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_distribution_shift_eval_protocol.json`
  when visual augmentation support is prepared
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/claim_boundary.json`
  when visual augmentation support is prepared
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_generation_run_manifest.json`
  when visual augmentation generation is run
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_generation_results.jsonl`
  when visual augmentation generation is run
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_generation_qa_manifest.json`
  when visual augmentation generation is run
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_training_readiness_manifest.json`
  when visual augmentation generation is run
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/visual_augmentation_training_dataset_manifest.json`
  when visual augmentation generation is run
- `robot_eval_jobs/<job_id>/oscar_visual_augmentation_packet/exports/visual_augmentation/episodes.jsonl`
  when visual augmentation generation is run
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

The `dev` extra carries the full no-GPU validation stack (`usd-core`/`pxr`,
`mujoco`, `trimesh`, `pycollada`/`collada`, `boto3`) so the dry-render,
scene-placement, and parity
tests run instead of skipping. For the canonical one-command CPU setup, the
import probe, and the green-baseline test commands, see
[docs/DEV_SETUP.md](docs/DEV_SETUP.md).

For the canonical non-secret production environment and the explicit boundary
between the current product path, local simulation, and paid-provider admission,
see [docs/PRODUCTION_PIPELINE_PROFILE.md](docs/PRODUCTION_PIPELINE_PROFILE.md).

Run repository commands through the synced environment:

```bash
uv run blueprint-capture-pipeline --help
```

This is a repository development setup only. It is not the supported single-VM GPU runtime bootstrap path.

Optional LLM support for the capture review agent:

```bash
uv sync --extra dev --extra llm
```

Local tests automatically add `src/` and the sibling `BlueprintContracts/src` to `sys.path` through [`tests/conftest.py`](tests/conftest.py). If the contracts repo is not present beside this repo, install `blueprint-contracts` before running `uv run pytest`.

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

With this profile, uploads without explicit requested outputs default into `qualification`, `evaluation_prep`, and `simulation_automation`; auto-staged `robot_eval_job_request.v1` work uses the MuJoCo runtime profile; and the control plane can drain accepted WebApp-style job requests into the packaged `blueprint_pipeline.mujoco_g1_simulator_command`. `BLUEPRINT_ALLOW_SIMULATOR_EXECUTION` remains an explicit gate, and a MuJoCo G1 asset root or `BLUEPRINT_MUJOCO_ALLOW_FETCH_G1_ASSETS=true` is required before the packaged command is configured. This proves only sim-only beta execution when the job artifacts contain trace, metric, visual media, and scenario-run coverage evidence. It does not prove generated-world rank fidelity or external robot-team closure. WAM/substrate outputs, when requested, add evaluator-bounded policy comparison only.

Local sim-only beta gate:

```bash
python scripts/run_sim_only_beta_local_gate.py \
  --capture-root /absolute/path/to/capture-root \
  --webapp-repo ../Blueprint-WebApp \
  --mujoco-g1-root /absolute/path/to/mujoco_menagerie/unitree_g1
```

This starts the real local Pipeline intake service with a synthetic token, runs WebApp forwarding preflight with the read-only intake probe, posts a WebApp-built `robot_eval_job_request.v1` through the WebApp route, processes the staged Pipeline inbox, runs the packaged MuJoCo sim-only command, and writes `pipeline/live_pipeline_control_plane/sim_only_beta_local_gate/sim_only_beta_local_gate_report.json`. The report must be `status=passed` before claiming local post-upload autonomy. The report remains local proof only; production forwarding, deployment parity, remote cloud execution, and generated-world rank fidelity require separate evidence.

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

The deployment/parity proof checks WebApp `/health/ready`, Pipeline intake `/health`, authenticated intake-audit reachability when the intake is routed under `/api/live-pipeline/*`, clean `HEAD == origin/main` repo parity, and deployed commit equality when commit values are supplied. A route-forwarding proof can supply the WebApp URL and forwarding endpoint URL when those fields are present, but deployed commits and the live intake token still come from deployment/runtime configuration. The release gate reads the local sim-only gate report and WebApp forwarding preflight report, then requires a current production route-forwarding proof for the same capture root plus deployment/parity proof before writing `pipeline/live_pipeline_control_plane/sim_only_beta_release_gate_report.json`. The report must be `status=passed` before claiming beta release readiness. generated-world rank fidelity and remote-cloud provider execution stay out of scope for this sim-only gate.

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
  --live-closure-evidence /path/to/live_eval_closure_evidence.json \
  --stage-webapp-request \
  --stage-arena-results \
  --stage-policy-package \
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
robot-team policy package references, delivery commands, closure evidence, or
live operator credentials are missing. The packet
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
If the request uses WebApp's public `/synced-artifacts/sites/<slug>` capture
root, set `ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT_BY_SITE_JSON` for that
slug or the single-site `ROBOT_EVAL_JOB_REQUEST_FORWARD_CAPTURE_ROOT` fallback;
the Pipeline inbox runner quarantines public roots without an explicit local
capture-root override.
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
reference blocked or incomplete. For an explicitly requested hardware-proof
diagnostic, add `--real-robot-pov` plus `--stage-real-robot-pov` to validate and
copy owner-supplied robot camera/action evidence to
`pipeline/robot_eval_inputs/real_robot_pov_manifest.json`. The control plane does
not require this for sim-only work; missing exact keys, camera/action evidence,
timestamp alignment, or owner evidence are recorded as real-POV diagnostics.
Generated POV storyboards remain support artifacts only. Add
`--deployment-outcomes` plus `--stage-deployment-outcomes` to validate and copy
job-specific actual pilot/deployment records into
`pipeline/robot_eval_inputs/<job_id>/deployment_outcomes/inbox/`; the robot-eval
job still has to pair those records with predictions before sim-vs-real
calibration is proven. Records with task/scenario IDs and actual-result signals
can be staged as real-world validation inputs before proof, but they are only
calibration-ready when each staged record includes `scenario_eval_run_id` or
`scenario_variation_instance_id` for an exact prediction join. Missing exact
join keys or owner evidence are recorded as calibration diagnostics, not
control-plane required inputs for sim-only work. Add
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

The forwarding token is not enabled by default because it is a live bearer
secret shared by the WebApp forwarding layer and the Pipeline intake service.
Do not hardcode it in source, generated reports, or checked-in env files. To
create a local ignored env file with matching WebApp and Pipeline variables:

```bash
python -m blueprint_pipeline.live_pipeline_forwarding_secret_setup \
  --env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env" \
  --forward-url "https://paperclip.tryblueprint.io/api/live-pipeline/job-requests" \
  --capture-root "$CAPTURE_ROOT" \
  --site-slug "$WEBAPP_SITE_SLUG"
```

Use that file on the Pipeline host before starting the intake service:

```bash
set -a
source "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"
set +a
blueprint-live-pipeline-intake-service --host 127.0.0.1 --port 8765
```

Use the same file from the WebApp repo for the read-only forwarding preflight:

```bash
npm run pipeline:forwarding:preflight -- \
  --require-forwarding \
  --probe-intake-audit \
  --forwarding-env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"
```

The sim-only beta deployment proof can read the same file:

```bash
python scripts/run_sim_only_beta_deployment_parity_proof.py \
  --capture-root "$CAPTURE_ROOT" \
  --route-forwarding-proof /absolute/path/to/production_route_forwarding_proof.json \
  --webapp-url https://www.tryblueprint.io \
  --pipeline-intake-url https://paperclip.tryblueprint.io/api/live-pipeline/job-requests \
  --forwarding-env-file "$HOME/.blueprint-secrets/live_pipeline_forwarding.env"
```

For production, install the same token value into the deployment secret store as
`ROBOT_EVAL_JOB_REQUEST_FORWARD_TOKEN` for the WebApp and
`BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN` for the Pipeline intake service. A local
env file alone cannot make a remote endpoint authenticate unless the remote
service has the same token configured.

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
prove generated-world rank fidelity.

For temporary internal demos, `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` allows the World Labs preview path to fall back to the raw walkthrough video when privacy processing is unavailable. The bypass path is intentionally labeled as non-production and unredacted, and the input video is auto-trimmed/compressed to World Labs upload limits before submission.

The production privacy deployment may use the privacy runner services:

- `sam3-detect`
- `vip-inpaint`
- `deepprivacy2-anonymize`

Legacy `video_to_world`, retrieval-index, and live-geometry validation material
remains in older docs for compatibility, but those paths are not part of the
active Capture App -> World Labs -> CPU preflight -> simulation-manifest flow.

The main `blueprint-pipeline` job stays CPU-only. The concrete service contract, storage behavior, and model-path rules are documented in [docs/PRIVACY_RUNNER_SERVICES.md](docs/PRIVACY_RUNNER_SERVICES.md).

The privacy path treats depth generation as a first-class optional artifact when
depth evidence is available or a depth runner is configured:

- use ARKit depth/confidence when available
- otherwise run Depth Anything 3 only when the depth runner is configured for the lane
- persist the resulting depth and confidence manifests for downstream grounding
- pass those manifests into VIP so non-ARKit inpainting reuses the generated depth artifacts

## Legacy GPU Bring-Up

For the current sample-video to owner-GPU proof path, use
[`docs/FIRST_GPU_E2E_RUNBOOK.md`](docs/FIRST_GPU_E2E_RUNBOOK.md).
It sequences local capture preflight, current pipeline lanes, WebApp forwarding,
owner GPU simulator command execution, proof ingestion, and closure audits
without promoting CPU or simulator smoke artifacts into rank-fidelity proof.
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

The older single-VM GPU runbook is a superseded historical record and is not an
active preview, upload, CPU-preflight, or simulation-manifest path. Use the
provider-specific runbooks and the shared `paid_resource_allocator` commands.

For privacy-service bring-up, use the service images under [`deploy/docker/`](deploy/docker) and the Terraform stack under [`deploy/terraform/main.tf`](deploy/terraform/main.tf).

The canonical no-GPU local setup is documented in
[`docs/DEV_SETUP.md`](docs/DEV_SETUP.md).
Use that setup before dry-render, USD placement, MuJoCo, or provider-staging
tests; it verifies `PIL`, `pxr`, `mujoco`, `trimesh`, `collada`, and `boto3`.

```bash
python -m blueprint_pipeline.cpu_env_doctor
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

The production post-capture producer is
`python -m blueprint_pipeline.post_capture_evidence_cli`. It verifies ARKit/Raw V3.2 bytes and
connects the existing reconstruction, geometry, automatic-target, placement,
scene-composition, routing, and policy-admission lanes into one
content-addressed run. It emits the first scientifically valid abstention when
an upstream qualification is unavailable. See
[`docs/runbooks/post-capture-evidence-spine.md`](docs/runbooks/post-capture-evidence-spine.md).

Provider-neutral new-site Task Evaluation Run compilation is available through
`python -m blueprint_pipeline.new_site_task_evaluation_run`. It binds admitted
capture/provider inputs, registered native 3DGS and dynamics geometry, a
deterministically authorized task target and robot placement, the best exactly
qualified task/site engine route, and five matched-reset learned-policy
attempts—or emits the smallest fail-closed abstention. See
[`docs/NEW_SITE_TASK_EVALUATION_RUN.md`](docs/NEW_SITE_TASK_EVALUATION_RUN.md).
Request v2 freezes a three-to-five-scenario inspection pack and executes the
same five learned policies across the complete grid. It retains missing and
failed cells, uses paired-scenario aggregation with deterministic uncertainty,
and surfaces ties, exclusions, unsupported metrics, and catastrophic failures.
The v1 single-reset reader remains supported.
The capture-materialization and explicit-task development acceptance command
`python -m blueprint_pipeline.new_site_task_evaluation` is an upstream,
non-ranking lane; its result cannot substitute for the five learned-policy
execution receipts required by the Task Evaluation Run compiler.

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
  --provider local
```

Optional agent-review wrapper:

```bash
blueprint-run-e2e \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider local
```

Use `--provider local` for deterministic no-LLM contract runs. Use
`--provider openai` or `--provider claude` only when the corresponding external
review provider is intentionally configured.

`blueprint-run-e2e` runs evaluation prep and the WebApp sync handoff by default
so the CLI matches the autonomous handoff listener. Use
`--skip-evaluation-prep` only for a narrow local developer run.

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
live `video_to_world`, simulator, or rank-fidelity proof.

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
[`docs/SIMREADY_ASSET_LANE.md`](docs/SIMREADY_ASSET_LANE.md).
It writes review artifacts only; it does not run Isaac Sim, MuJoCo, PyBullet,
live providers, model downloads, or rank-fidelity trials.
Evaluation prep surfaces existing SimReady artifacts but does not auto-build
them unless `BLUEPRINT_ALLOW_LEGACY_SIMREADY_EVAL_PREP=true` is set.

Isaac/G1 kitchen-parity render support:

```bash
python scripts/run_isaac_g1_kitchen_parity_eval.py \
  --request /path/to/request.json \
  --kitchen-usd /path/to/Collected_KitchenRoom/KitchenRoom.usd \
  --out-dir /tmp/blueprint-g1-dry-render \
  --dry-render
```

`scripts/run_isaac_g1_kitchen_parity_eval.py` owns the G1 kitchen-parity review
lane, including the head-POV `open the refrigerator` seed path and the local
`--dry-render` preview. Dry-render is CPU-only support evidence for
stance/camera/arm-framing checks. It is not a rendered Isaac frame, policy
success, physical object contact, safety validation, deployment approval,
learned-policy success, or live robot readiness.

Recent G1 support modules:

- `src/blueprint_pipeline/scene_placement/` provides pure, swappable placement
  helpers. Importing the package pulls in no `isaacsim`, `torch`,
  `google-genai`, network, or GPU dependency; USD, perception, VLM, and PhysX
  backends are injected. See
  [`src/blueprint_pipeline/scene_placement/README.md`](src/blueprint_pipeline/scene_placement/README.md).
- `src/blueprint_pipeline/warm_render_server.py` implements a hermetically
  tested warm-serve control loop. Live multi-job reuse after one real Isaac
  scene load still requires on-GPU proof.
- `src/blueprint_pipeline/provider_race.py`, `src/blueprint_pipeline/render_lock.py`,
  and `scripts/gpu_spend_guard.py` are provider/spend-safety scaffolding. They
  do not prove live provider execution, teardown, artifact quality, or readiness
  by themselves.

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
[`docs/PALATIAL_PHYSREADY_LANE.md`](docs/PALATIAL_PHYSREADY_LANE.md).

Legacy local Marble sim-asset handoff module:

```bash
PYTHONPATH=src python -m blueprint_pipeline.marble_sim_assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Use `--world-manifest /path/to/worldlabs_world_manifest.json` to review an
explicit local World Labs world manifest. The Marble handoff lane is documented
in
[`docs/MARBLE_SIM_ASSET_HANDOFF.md`](docs/MARBLE_SIM_ASSET_HANDOFF.md).
It reads persisted World Labs manifests and emits Isaac Sim, MuJoCo, and
PyBullet review packets without downloading remote assets, calling World Labs,
running simulators, or claiming generated-world rank fidelity.
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
generated-world rank fidelity.

Scaniverse-assisted asset import:

```bash
blueprint-import-scaniverse-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --asset /path/to/scaniverse-export.usdz \
  --asset /path/to/scaniverse-splat.ply \
  --blueprint-sidecar /path/to/blueprint-scaniverse-sidecar.json
```

This stages local Scaniverse exports into `pipeline/scaniverse_assets/`,
preserves checksums/source metadata, and forwards CPU-preflight-supported local
formats into scene-asset preflight. It does not call Niantic APIs, run
simulators, or treat Scaniverse geometry as raw capture truth. The lane is
eligible for PTDP packaging only as an `external_derived_support_asset`, so
buyer readouts must keep it separate from raw Blueprint evidence, task success,
physics/contact proof, and deployment readiness. The lane is
documented in
[`docs/SCANIVERSE_ASSET_IMPORT.md`](docs/SCANIVERSE_ASSET_IMPORT.md).

Polycam Developer Mode raw-ZIP source-profile binding:

```bash
blueprint-adapt-polycam-developer-raw \
  --archive /path/to/polycam-developer-raw.zip \
  --declaration /path/to/polycam-source-declaration.json \
  --output /path/to/polycam-source-profile.json \
  --source-commit-sha <40-hex-immutable-commit>
```

This local-only adapter preserves and hashes the original ZIP, hashes every
regular archive member, and binds declared RGB/video, timestamps, intrinsics,
extrinsics, depth, confidence, mesh, metric-unit, capture, device, and provider
lanes. It rejects unsafe archives and emits a fail-closed abstention when a
semantic lane is missing. The result is Polycam-derived support, not Blueprint
Raw Contract truth or independent metric/collision/Isaac/task evidence. See
[`docs/POLYCAM_DEVELOPER_SOURCE_PROFILE.md`](docs/POLYCAM_DEVELOPER_SOURCE_PROFILE.md).

To also download local PLY/SPZ splats for object-index enrichment:

```bash
blueprint-materialize-worldlabs-assets \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --include-visual-assets
```

Optional Splat Analyzer object-index backend:

```bash
export SPLAT_ANALYZER_REPO=/opt/splat_analyzer
# or:
export SPLAT_ANALYZER_RUN_LOCAL=/opt/splat_analyzer/run_local.py
# or provide a custom command template:
export SPLAT_ANALYZER_COMMAND='python /opt/splat_analyzer/run_local.py --ply {SPLAT_PATH} --prompt {PROMPT} --quality medium --job_dir {JOB_DIR}'

python -m blueprint_pipeline.object_index_stage \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --force-rebuild
```

The object-index stage runs `scripts/object_index_splat_analyzer_runner.py`
after the existing YOLO-World, Grounding-DINO, and SAM3 backends. The runner
discovers local `.ply` or `.spz` splats from
`pipeline/worldlabs_assets/materialized_assets_manifest.json`,
`pipeline/worldlabs_export_manifest.json`, `pipeline/worldlabs_world_manifest.json`,
`pipeline/simulation_automation/scene_asset_inventory.json`, or local scans.
It writes backend IO under `raw/object_index_artifacts/` and normalizes
Splat Analyzer `interactions.json` output into `raw/object_index.json` objects,
`raw/object_index_build_report.json` provider counts, and advisory
`raw/object_grounding_hints.json.scene_relationship_candidates`.

Those objects and relationships are model-derived candidates for task creation,
object geometry prep, and reviewer triage. They are not raw capture truth,
collision/contact proof, articulation-state proof, robot spawn validation,
simulator execution proof, policy execution proof, physical-robot readiness, or
public-claim support.

Fail-closed simulation automation plan:

```bash
blueprint-run-simulation-automation \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

The simulation automation lane is documented in
[`docs/SIMULATION_AUTOMATION_LANE.md`](docs/SIMULATION_AUTOMATION_LANE.md).
It writes local orchestration manifests only, including an optional
`isaac_lab_arena` Arena Pack review packet. It does not run simulators, download
assets, start training, call providers, or prove generated-world rank fidelity unless explicit
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
prove generated-world rank fidelity, generated-world rank-fidelity result, off-scope approval, public
readiness, or customer-specific SRCC.

The intended WAM/substrate proof target is narrow: compare policy A against
policies B and C inside the configured evaluator over the same scenario matrix,
observation protocol, and scoring protocol. This follows the OSCAR / SC3-Eval
style of using generated or simulated evaluator rollouts to preserve policy
rankings and measure correlation with real anchors when such anchors exist. A
passing `policy_ranking_scorecard.json` can say which policy ranked higher in
that evaluator. It cannot say the policy is deployable, safe, physically ready,
or generally superior outside the evaluator. Metrics such as MMRV, Spearman, and
Pearson belong to calibration against paired real-world anchors, not to a
default sim-only or fixture-only run.

The preferred new learned-WAM evaluator candidate is `cosmos3_wam`
(Cosmos3-Nano) when a real adapter, checkpoint/provider runtime, and explicit
run gates are configured. This is not a permanent company dependency and not
proof of universal grading. `oscar_wam` remains the OSCAR baseline/compatibility
lane: OSCAR fine-tunes `Cosmos-Predict2.5-2B` on 180,657 filtered episodes
(94,830 robot and 85,827 human egocentric) and reports skeleton-conditioned
RoboArena policy-eval MMRV 0.571, Spearman 0.750, Pearson 0.852, and OSCAR
`success_rate_difference_pp` 1.73. Its GPT-5 success scorer matched 78/100
human labels, had specificity
0.90, and missed about one third of real successes. `cosmos_wam` remains a
legacy/advisory Cosmos-Predict2.5 baseline; NVIDIA's Cosmos-Predict2.5 repo says
future releases, docs, and community support are focused on Cosmos 3.

For learned `oscar_wam` execution, Blueprint uses the official public release
path: `https://github.com/wuzy2115/oscar-public.git` pinned to
`4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`, with
`zywu2115/OSCAR-2B` pinned to HF revision
`c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`. The
`zywu2115/OSCAR_policy_rollout` dataset is reference data, not runtime code.
Provider bundles, the reusable OSCAR GPU image, and local learned OSCAR
adapters record an `official_oscar_release` contract; unpinned local source,
checkpoint, or provider images block learned OSCAR claims unless explicitly
marked experimental.

SC3-Eval is treated as a recipe: forward/inverse dynamics consistency,
cross-view consistency, and uncertainty-driven early termination are
reliability/abstention signals, not task-success labels or rank-fidelity proof.
SC3-Eval initializes from Cosmos3-Nano and reports headline closed-loop Pearson
0.929 / MMRV 0.119. Its in-distribution online split is 0.984 / 0.022 versus
Cosmos-Predict2.5 at 0.897 / 0.090; on the out-of-distribution online split,
Pearson is 0.870 versus 0.871 while MMRV is better at 0.171 versus 0.195. Its
published scope is 381 hours in one table-bussing scene, 12 object categories,
three camera views, seven policy checkpoints, and at most 20-second rollouts.
These are SC3-Eval paper results, not Blueprint measurements; Blueprint has not
measured equivalent rank fidelity.
Blueprint writes `sc3_eval_protocol.json` to keep those requirements, accepted
anchor joins, correlation metrics, and robot/policy adapter contracts explicit.
See
[`docs/SC3_EVAL_PROTOCOL.md`](docs/SC3_EVAL_PROTOCOL.md).
For benchmark-grade comparisons across evaluator backends, use
`python -m blueprint_pipeline.benchmark_protocol compile` and
`python -m blueprint_pipeline.benchmark_protocol report`. That protocol
adds frozen public/hidden splits, fixed non-replaceable attempts, exact baseline
and candidate checkpoint digests, seen/unseen generalization slices, confidence
intervals, digest-bound episode evidence, and scoped external rank comparison.
See
[`docs/BLUEPRINT_BENCHMARK_PROTOCOL.md`](docs/BLUEPRINT_BENCHMARK_PROTOCOL.md).
`cosmos3_super` is a high-cost adjudication candidate, not the default local
path. NVIDIA released the 4B `cosmos3_edge` model on 2026-07-20, but Blueprint
does not treat it as a default or qualified runtime. Edge requires a distinct
model profile and Blueprint-specific runtime/ranking evidence; it must not
inherit the Nano-specific SC3-Eval recipe or paper results. See
[`docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md`](docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md).

### NVIDIA SIGGRAPH 2026 experimental support lanes

The SIGGRAPH integrations are replaceable, advisory sidecars. None run by
default, none install prerelease NVIDIA packages into the core environment, and
none upgrade simulator, task-success, policy, ranking, real-sensor, or
deployment claims.

- `python -m blueprint_pipeline.external_simready_validation` normalizes a pinned SimReady
  Foundation profile report. Use
  `scripts/setup_simready_validator_env.sh` and
  `scripts/run_simready_validator_worker.py`; transformations are prohibited in
  the v1 lane.
- `python -m blueprint_pipeline.simready_rule_calibration` compares repeatable validator findings
  with frozen expert labels. Only explicitly approved zero-error rules may be
  promoted to a CPU/pre-GPU blocking gate; all others remain advisory.
- `python -m blueprint_pipeline.omniverse_library_preflight ovrtx` and
  `python -m blueprint_pipeline.omniverse_library_preflight ovphysx` launch
  isolated workers through an explicit command template. They require both the
  CLI flag and `BLUEPRINT_ALLOW_OMNIVERSE_EXTERNAL_PREFLIGHT=true`, record cold
  and warm runs, and write request/result/runtime-receipt/claim-boundary
  artifacts under `pipeline/sensor_preflight/` or
  `pipeline/physics_preflight/`. The reference workers are
  `scripts/run_ovrtx_preflight_worker.py` and
  `scripts/run_ovphysx_preflight_worker.py`.
- An ovrtx preflight automatically adds specific checks for
  `ParticleField3DGaussianSplat`, time-sampled episode state, semantic ID maps,
  lidar/radar structure, and configured robot/target visibility. A generic USD
  load cannot satisfy those checks.
- `python -m blueprint_pipeline.omniverse_library_preflight benchmark` retains a library only when
  the exact scene also has accepted Isaac execution evidence and the sidecar
  demonstrates repeatable outputs, relevant failure coverage, preserved sensor
  metadata, and the configured cold/warm runtime advantage.
- `python -m blueprint_pipeline.omniverse_library_preflight benchmark-suite` additionally requires
  valid and intentionally negative same-scene fixtures, CPU/GPU memory
  measurements, and matching Isaac failure detection.
- `python -m blueprint_pipeline.cosmos3_edge_experiment` is a separate 4B Edge profile; it
  requires a frozen privacy-safe cell manifest and runs forward, inverse, and
  reasoning modes separately. It never inherits Cosmos3-Nano/SC3 qualification.
  NVIDIA's July 20 model card does not list Unitree G1 7D actions among the
  supported action encodings, so the adapter blocks that substitution.
- `python -m blueprint_pipeline.cosmos3_edge_qualification` consumes the Edge attempt
  manifest, validated evaluator runtime receipt, and a frozen Blueprint
  scorecard. It measures grounding, abstention, rank correlation, and failure
  recall while still requiring a separate owner decision for any default.
- `python -m blueprint_pipeline.gsplat_conformance` compares Blueprint's current
  `ParticleField3DGaussianSplat` authoring with a pinned
  `usd-convert-gsplat` oracle, including array values, quaternion sign
  equivalence, stage units and up axis. It does not replace the current author.
- ArtiFixer output is generated support pending a frozen, disjoint held-out
  real-view evaluation. It remains neither captured pixels nor collision
  geometry.
- `python -m blueprint_pipeline.nvidia_siggraph_policy` writes the component policy registry.
  Content Agents, SimReady Blender, standalone ovstage, and conference research
  systems remain deferred behind their component-specific blockers. A fresh
  source/version/license review is mandatory on or after 2026-07-24.
- `python -m blueprint_pipeline.nvidia_asset_conditioning_review` records immutable proposal-only
  evidence for buyer CAD, Content Agent, or Blender workflows. Physical
  metadata stays non-authoritative and Content Agent/Blender proposals require
  a human approval receipt.
- `python -m blueprint_pipeline.nvidia_experiment_resource` binds paid execution to the
  shared allocator admission and requires exact-attempt, global inventory,
  zero-burn, and billing reconciliation evidence.
- `python -m blueprint_pipeline.nvidia_siggraph_completion` maps every memo lane to
  source, schema, test, and verification evidence while leaving external GPU
  qualification explicitly unproven.

Paid execution must enter through
`python -m blueprint_pipeline.paid_resource_allocator cpu-build`,
`model-volume`, or `gpu-canary`; provider-specific launchers are not supported.
The experimental commands do not allocate resources themselves. Simulation
automation merely surfaces any resulting artifacts as advisory inputs.

When `eval_ready_task_grounding.json` is present, the OSCAR/Cosmos WAM evaluator
copies it into the job directory, enriches task prompts with the selected
task-object target, attaches the camera calibration quality gate, consumes the
robot FK/projected-skeleton trace as action-conditioning support, records the
lightweight articulated-object proxy check, and writes
`wam_prediction_outcome_correlation_ledger.json`. These files ground and audit
the learned rollout, but they stay support artifacts: calibration gates,
projected skeletons, VLM labels, and handle proxies do not prove physical
contact, torque, safety, generated-world rank-fidelity result, or real-world task success.

The MuJoCo Unitree policy/WAM closed-loop helper now has a default local
OSCAR-style support backend for the no-live-provider case. If no gated
OSCAR/Cosmos WAM command is configured, `run_robot_policy_wam_closed_loop_attempt`
generates action-conditioned next-observation frames plus short MP4 segments,
records the policy action, simulated proprioception keys, and projected G1
skeleton support in `wam_generated_next_observations.jsonl`, then re-queries
the selected Unitree policy on those generated frames. Those artifacts are labeled
`default_local_wam_generator_used=true` and
`learned_oscar_or_cosmos_model_ran=false`; they are useful loop evidence, not a
claim that a learned OSCAR/Cosmos checkpoint ran. This fallback is Blueprint's
deterministic support implementation for tests and local no-provider runs; it
is not the official OSCAR release and must not upgrade provider/runtime proof.
The loop also writes the WAM-derived perception/observation harness artifact
family under `robot_policy_wam_closed_loop/wam_derived_observation_harness/`.
The harness derives support masks/boxes, tracks, relative depth, pose, contact
likelihood, reviewability, uncertainty, and adapter reports from generated media
plus evaluator-controlled state. It passes only declared policy fields back to
the policy and keeps masks, depth, contact likelihood, and uncertainty as
diagnostic/scoring-gate support unless the policy declares those fields.
Optional detector/depth/tracking backends are configured through
`BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_KIND`,
`BLUEPRINT_WAM_PERCEPTION_HARNESS_BACKEND_COMMAND`, and
`BLUEPRINT_ALLOW_WAM_PERCEPTION_HARNESS_EXTERNAL_BACKEND=true`; otherwise the
fixture backend runs and external backends fail closed. When labeled
capture-backed validation rows are supplied to the harness API, it writes
validation metrics and false-success reduction metrics comparing plain generated
video labels against harness-gated scoring. Without those rows, those metrics
remain explicitly blocked/not measured. The markdown review report summarizes
per-step confidence, withheld policy fields, validation status, and claim
boundaries for customer or operator review.
Inferred depth is not sensor depth, masks are not physical truth, and contact
likelihood is not physical contact proof.

To run the sim-only provider/harness loop proof, use:

```bash
python -m blueprint_pipeline.wam_sim_provider_e2e \
  --provider-mode real \
  --generated-frame <gpt-image2-or-wam-generated-frame.jpg> \
  --target-prompt "robot arm" \
  --sam3-weights <sam3.pt> \
  --depth-provider v2 \
  --pose-model yolo11n-pose.pt
```

This runner proves the architecture path only: generated frame in, SAM3/depth/
pose providers through the replaceable backend path, harness artifacts out,
policy adapter field gating, and claim-bounded scoring/requery status. It also
works with `--provider-mode fixture` for deterministic tests. If no
`--generated-frame` is supplied, it discovers an existing generated frame under
`robot_eval_jobs/` or writes a local synthetic AI-style start frame. The manifest
records that optional truth-label validation was not requested; labeled
validation rows are not required and do not block this sim-only proof.

The default depth provider for this smoke path is Transformers Depth Anything
V2 small. Depth Anything 3 is optional and selectable with
`--depth-provider da3`, `BLUEPRINT_ALLOW_WAM_AUTO_DA3_PROVIDER=true`, and
`BLUEPRINT_WAM_DA3_MODEL_ID` such as `depth-anything/DA3-BASE`. Missing DA3
runtime or weights block that provider path explicitly; generated-pixel depth is
still not sensor depth.

To avoid reinstalling the harness provider stack on every new GPU run, generate
a reusable WAM perception harness image context:

```bash
blueprint-build-wam-perception-harness-gpu-image \
  --image-ref docker.io/nijelhunt/blueprint-wam-perception-harness:20260626-cu126
```

The generated context writes `Dockerfile.wam-perception-harness-gpu`,
`build_image.sh`, `push_image.sh`, `run_image_healthcheck.sh`,
`prepare_model_mounts.sh`, and
`wam_perception_harness_gpu_image_manifest.json`. The image bakes Blueprint
harness code, PyTorch/CUDA, `transformers`, `ultralytics`, Depth Anything V2
cache, and YOLO pose cache. It does not bake Docker, DigitalOcean, Hugging Face,
or object-store tokens. SAM3 weights are expected at `/models/sam3/sam3.pt`
through a mount or provider-side model fetch, and the manifest records
`bakes_sam3_weights=false`. The image build, push, or healthcheck still does not
prove perception accuracy, sensor depth, physical contact, off-scope validation, or
generated-world rank fidelity.

To test the real-provider lane without weakening those boundaries, run:

```bash
python -m blueprint_pipeline.wam_real_provider_validation_probe run \
  --generated-frame <wam-generated-frame.jpg> \
  --validation-set <capture-backed-validation-rows.json>
```

The probe writes `wam_real_provider_validation_proof_manifest.json` plus the
normal harness artifact family under
`robot_eval_jobs/wam_real_provider_validation_probe_<timestamp>/`. It checks for
SAM3 weights (`SAM3_WEIGHTS_PATH` or `BLUEPRINT_SAM3_WEIGHTS_PATH`), optional
depth and pose commands (`BLUEPRINT_WAM_DEPTH_PROVIDER_COMMAND`,
`BLUEPRINT_WAM_POSE_PROVIDER_COMMAND`), and labeled validation rows. Missing
providers or labels produce a blocked manifest, recommend early termination,
block policy requery/success scoring, and leave perception accuracy
`not_measured`; that blocked result is the correct behavior until real provider
outputs and labeled validation data are present.

The validation set is treated as real only when rows are capture-backed or
accepted real-world anchors, include an actual label such as `actual_success`,
`capture_success`, `expected_target_visible`, `expected_contact`, or
`expected_object_id`, and carry a source reference such as `source_capture_path`,
`source_artifact_path`, `source_video_path`, `source_frame_path`,
`source_label_path`, `evidence_path`, or `operator_attestation_path`. Fixture
rows can test the contract, but they do not satisfy the real-provider proof.

Forward/inverse episode consistency is a separate scorer layer, not a property
claimed by WAM execution or by the evaluator itself. The OSCAR/Cosmos WAM
evaluator writes `wam_episode_consistency_request.json`; a separate VLM or human
review command writes `wam_episode_consistency.command.json`; the evaluator then
normalizes that result into `wam_consistency_checks.json`. See
[`docs/WAM_EPISODE_CONSISTENCY_SCORER.md`](docs/WAM_EPISODE_CONSISTENCY_SCORER.md).

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
official/trusted checkpoint, generated-world rank fidelity, generated-world rank-fidelity result, safety
validation, or correctly placed-object task success.

Do not use OpenVLA, OSCAR, Cosmos, fixture WAM, or generated WAM rollouts as the
G1 robot policy. OpenVLA can remain a generic comparison candidate for
non-Unitree work, and WAM outputs remain evaluator/support artifacts unless a
separate Unitree-specific policy endpoint consumes the observation and emits
normalized G1 actions. For this machine, source `.env.unitree.local` to bind the
verified local Unitree RL Gym root and checkpoint before running G1 MuJoCo policy
proofs. See
[`docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md`](docs/UNITREE_G1_POLICY_ENDPOINT_LANE.md).

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

The intended fresh endpoint architecture is still Unitree-native: a canonical
allocator-owned worker must launch a long-lived UnifoLM `/act` server, then the
local endpoint can call it through
`blueprint-unitree-unifolm-vla-server-bridge --server-url
https://<pod_id>-8777.proxy.runpod.net/act`, or use an equivalent Unitree
LeRobot/GR00T-SONIC command endpoint. The legacy public Unitree RunPod launch
mode is hard-disabled until it is routed through `paid_resource_allocator`.
OpenVLA remains only a comparison
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

For learned-WAM review-quality rollouts, run the short visual sanity gate before
attempting longer autoregressive loops:

```bash
blueprint-run-persistent-wam-short-visual-sanity \
  --policy-observation /path/to/policy_observation.json \
  --provider vast \
  --transition-count 2
```

The command first runs source policy-observation visual QA locally. If that QA
fails, no provider runner starts. When it proceeds, it forces review-quality WAM
settings for one or two transitions, writes
`persistent_wam_short_visual_sanity_manifest.json` plus the source QA report,
WAM visual-quality report, contact sheet, frame stats, review-video status, and
ffprobe metadata. If RunPod or Vast is used, the manifest also records teardown
status and `continuing_spend_from_this_run`. Longer review-quality WAM rollouts
must set `BLUEPRINT_PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST` to a passed
short-sanity manifest for the same policy observation; the legacy
`BLUEPRINT_ALLOW_PERSISTENT_WAM_LONG_REVIEW_ROLLOUT` flag alone is not enough to
unlock the long run. These artifacts prove only reviewability of model-derived
support media, not task success, safety, generated-world rank fidelity, or raw
capture truth.

For paid Vast runs, keep startup/log transport bounded separately from WAM
rollout runtime. `BLUEPRINT_VAST_HEARTBEAT_NO_PROGRESS_SECONDS` controls how
long the Vast adapter waits for onstart/request_logs progress before writing
`vast_heartbeat_no_log_progress_timeout`; Unitree GR00T/SONIC persistent runs can
override it with
`BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HEARTBEAT_NO_PROGRESS_SECONDS`.
`BLUEPRINT_VAST_WAM_NO_PROGRESS_SECONDS` remains the longer WAM/provider runtime
watchdog and should not be used to justify a silent startup instance.
Isaac G1 parity Vast render jobs default to a `$5/hr` offer cap; override with
`--vast-max-hourly-rate <usd>` or
`BLUEPRINT_ISAAC_G1_PARITY_VAST_MAX_HOURLY_RATE` for a specific run.

Synthetic fallback initial observations and synthetic 2D WAM seeds are blocked
from live or review-quality WAM provider bundles by default. Use
`BLUEPRINT_ALLOW_SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT=true` only for an
explicit experiment; emitted bundle, runtime, and review artifacts must still
label `capture_truth=false`, `geometry_truth=false`, and
`visually_useful_rollout` separately from provider/runtime success.

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
a provider-launchable input. Isaac RunPod requests also block before spend with
`prebuilt_isaac_eval_worker_image_ref_missing` when no prebuilt Isaac worker
image is configured. The raw Isaac Sim base image is not the fast production
path because it can spend bounded startup time without reaching the Blueprint
fetch/finalizer wrapper or uploading `isaac_provider_runtime_output.zip`; only
use `BLUEPRINT_ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD=true` for an intentional
debug run.
RunPod GPU allocation is supported only through
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...`; that
allocator requires and binds one-resource/spend evidence plus an independently
armed watchdog. Terminal provider absence still requires watchdog/closure
evidence. The legacy live-proof collector and provider-adapter mutation modes
are hard-disabled as public launch paths and remain available only for
dry-run/read-only or allocator-owned internal use. A missing first output zip records
`provider_pod_startup_or_image_pull_timeout` and can still prove teardown, but
it is not Isaac Sim execution proof; an empty zip left by a staging PUT probe is
rejected. The canonical allocator binds an image/container startup canary, or
the closed `--probe-kind strict-policy-smoke` three-action GR00T/SONIC probe,
to the same launch request and its protected release evidence. The strict
policy probe uses a fixed adapter-owned command, ignores caller-provided
commands, uploads an exact three-action result, and does not claim Isaac task
success or physical robot control. If the startup canary
times out, the blocker is before user-command
execution; artifacts record `image_startup_canary_artifact_timeout` and, when
image metadata shows oversized layers, `prebuilt_isaac_image_layer_pull_exceeded_watchdog`.
Set `BLUEPRINT_RUNPOD_IMAGE_STARTUP_CANARY_HOLD_SECONDS=<seconds>` only when you
intend to keep the canary pod warm briefly for an immediate same-host reuse
attempt, and still collect teardown proof afterward. If image-size metadata
already shows an oversized worker layer, fresh `on-demand-pod` launches block
before spend with `large_worker_image_requires_canary_or_warm_provider` unless
`BLUEPRINT_ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START=true` is set for an intentional
debug retry.
Prefer S3-compatible object-store staging for the canary bundle and callback
URL, for example `blueprint-stage-wam-provider-object-store --job-dir
<job-dir>/object_store_canary --bundle-path <job-dir>/isaac_provider_runtime_bundle.zip`.
The helper name is historical; the signed GET/PUT transport is
simulator-agnostic. Export the signed PUT value with `$(cat
<provider_output_put_url.txt>)` or another shell-safe assignment instead of
unquoted sourcing, because presigned URLs contain `&` separators.
The outer fetch/upload wrapper uses `BLUEPRINT_ISAAC_PROVIDER_PYTHON`
when set, otherwise a normal `python3`/`python`, and falls back to
`/isaac-sim/python.sh` only when needed so first phase uploads do not
intentionally wait on Isaac Sim Python bootstrap. Direct stopped-pod restart
through the legacy adapter is disabled; a future reuse flow must be admitted
and owned by the canonical allocator. Provider host-capacity errors remain
separate from Isaac runtime proof.
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

Only bounded image/startup canaries and the fixed three-action learned-policy
smoke are currently supported through the canonical allocator. A
`gpu_provider_launch_request.json` at
`request_manifest_ready` is not authority to run a production robot evaluation:
general paid provider eval, Vast, and Lambda execution remain disabled until
each has a canonical allocator route. The old arbitrary provider-command and
provider-race launchers cannot invoke a subprocess or provider API.

```bash
python -m blueprint_pipeline.paid_resource_allocator gpu-canary \
  --provider-launch-request "$CAPTURE_ROOT/pipeline/robot_eval_jobs/$ROBOT_EVAL_JOB_ID/gpu_provider_launch_request.json" \
  --release-evidence /path/to/protected-release-evidence.json \
  --model-cache-evidence /path/to/verified-model-cache-evidence.json \
  --preflight-bundle /path/to/provider-preflight-bundle.json \
  --expected-image-source-commit <source-commit-recorded-by-release-evidence> \
  --provider-output-put-url-file /path/to/private-runtime-output-put-url.txt \
  --admission-out /path/to/gpu-canary-admission.json \
  --bound-request-out /path/to/bound-runpod-request.json \
  --adapter-output /path/to/runpod-provider-result.json \
  --pod-name <watchdog-bound-pod-name-from-preflight> \
  --execute
```

For the model-backed learned-action smoke, add
`--probe-kind strict-policy-smoke`. This remains a bounded policy-action probe;
general paid robot evaluation and dynamic episode execution are still disabled
until their own fixed canonical route exists.

Omit `--execute` for admission/binding validation without a provider mutation.
The release evidence must carry the same full image source commit supplied through
`--expected-image-source-commit` (`--expected-source-commit` remains a migration
alias). The preflight observation must be no more than five
minutes old, and the signed output URL must be in a regular `0600` file. Its
value is injected only for the adapter call and is never copied into admission
evidence. The allocator independently derives and records a clean
`orchestrator_source_commit`; it does not require that control-plane commit to
equal the immutable image's `image_source_commit`. Local and remote `main`
pointers remain recorded diagnostics rather than runtime identity. Admission
still fails closed on a dirty checkout, mutable image refs, mismatched runtime
inputs, model/base-image refs, dependency-lock evidence, or signed runtime
overlay digests. Existing v1 release/session evidence remains readable for
status, collection, refresh, and teardown migration.
The RunPod adapter remains useful for dry-run request-shape inspection, but its
public paid modes exit with the stable legacy-disabled blocker. Provider
allocation alone never proves simulator execution, generated-world rank
fidelity, safety, or public-claim upgrades.
For Lambda Cloud, the repo-owned adapter command is
`blueprint-run-lambda-provider-adapter`. It targets Lambda Cloud On-Demand Cloud,
not AWS Lambda. Dry-run mode writes `lambda_provider_adapter_result.json`,
`lambda_provider_readiness_manifest.json`, and
`provider_worker_endpoint_manifest.json` without an API call. Live API modes
are hard-disabled until a Lambda allocation path is routed through the canonical
allocator. Read-only inventory modes remain available. The adapter follows the
official Lambda Cloud API shape: Bearer auth against
`https://cloud.lambda.ai/api/v1`, `POST /instance-operations/launch` with
`region_name`, `instance_type_name`, and exactly one `ssh_key_names` entry, and
`POST /instance-operations/terminate` with `instance_ids`. Use `--mode
list-instances`, `list-instance-types`, `list-ssh-keys`, `list-images`, or
`list-regions` for no-spend account inventory after the API key is installed.
A submitted Lambda launch is only provider allocation submission; it is not
worker `/readyz`, simulator execution, artifact upload, teardown, safety,
physical-robot readiness, or rank-fidelity proof. Terminate Lambda instances via
the Lambda API/console and then run a list-instances follow-up; OS-level
shutdown is not accepted as spend closure because Lambda documents that billing
can continue.

Each job also writes `gpu_cost_control_ledger.json` with requested budget,
maximum billable GPU seconds, max workers, timeout, idle-shutdown/watchdog
requirements, concrete idle timeout, concrete external watchdog TTL, estimated
GPU seconds, actual GPU seconds when owner-runtime evidence exists, and the
blockers preventing allocation. A blocked scheduler or missing provider gate
records zero estimated GPU seconds and no live provider calls.

Run `blueprint-audit-provider-closure --job-dir
<capture-root>/pipeline/robot_eval_jobs/<job_id>` to verify optional provider
closure artifacts without provider API calls. The report checks local watchdog,
spend-ledger, artifact-output finalizer/upload, and teardown evidence, then
writes `provider_closure_audit_report.json`. Missing credentials or provider
artifacts are recorded as `blocked_optional_provider_closure`. This audit is
not required for local sim-only beta and does not prove rank fidelity, physical
readiness, safety validation, or field success.

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
closure coverage checks, and explicitly requested evidence exports inside a
Task Evaluation Run use that matrix
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
generated-world rank fidelity, off-scope validation, or public claims without separate
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

The historical Policy Improvement Run builder is deprecated compatibility
machinery, not a product or default orchestration path. If explicitly invoked
as an internal candidate-generation experiment, it may diagnose failures,
prepare curricula, or test a candidate, but only a new Task Evaluation Run may
make decision claims and the frozen policy-ranking result remains
`thesis_not_supported`. The legacy contract stays readable and
customer-supplied-policy friendly:
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
`private_asset_hosted_by_blueprint`; if the customer keeps execution in its own
environment, use `owner_evidence_bridge` and require camera/action/outcome
evidence joined to exact `scenario_eval_run_id` values.

```bash
python -m blueprint_pipeline.policy_improvement_run \
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
`private_hardware_integration_plan.json`, `rl_post_training_handoff_packet.json`,
`policy_improvement_run_offer.md`, and `policy_improvement_run_webapp_summary.json`
under `policy_improvement_run/`.
The manifest binds together the scenario matrix, normalized baseline attempts,
standard evaluation scorecard projection, failure labels, Post-Training Data
Package export, policy-autoresearch candidate package, heldout result, staged
readiness ladder, RL/post-training handoff packet, WebApp-safe summary projection,
and proof boundary. The handoff packet carries the success definition, sparse
reward signal, recoverable failure labels, intervention labels, timing/throughput
metrics, policy baseline fingerprint, same-condition frozen-baseline A/B plan,
bottleneck stage detection, speed curriculum plan, action-chunk continuity QA,
and intervention/safety ledger. It can say the run is ready for baseline
evaluation, failure diagnosis, post-training package build, policy-autoresearch,
candidate promotion, or customer review. It cannot turn sim heldout success into
generated-world rank-fidelity result: sealed audit scenarios must remain outside
training, old-run-only comparisons are not enough for candidate-improvement
claims, and generated-world rank fidelity, physical off-scope validation,
policy-ranking outcome, safety validation, and public claim upgrades remain false
until separately proven by accepted live evidence.

When `--arena-results-dir` points at existing Isaac Lab-Arena rollout artifacts,
the job ingests those local results into normalized traces, labels, clips,
metrics, reports, delivery manifests, rerun queues, and a Post-Training Data
Package. That proves package code paths and result ingestion only; simulator
execution, robot policy success, contact/off-scope validation, and generated-world rank fidelity
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
privacy scope, live simulator execution, live policy execution, optional
real-world outcome diagnostics, optional predicted-vs-actual calibration, review
acceptance, signed delivery, and optional safety/contact/physics diagnostics.
Missing optional real-world outcome, real-POV, calibration, or
safety/contact/physics evidence does not block sim-only closure and does not
upgrade real-world proof. Scenario-library and
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
status, policy-ranking outcome status, predicted-vs-actual status, and proof booleans
match the report. A section-complete report stub is not enough.
The simulator-engine plugin gate requires every supported engine in
`simulator_engine_plugin_registry.json` to have a ready adapter contract and
managed execution support; a partial or blocked registry remains a closure
blocker. Predicted-vs-actual closure records diagnostics when deployment outcome
records carry run-level identifiers that do not match a prediction. Real-world
validation closure recomputes owner evidence and actual-outcome signals from
each ledger row as optional proof evidence; aggregate `real_world_outcome_proven`
booleans alone cannot upgrade a real-world claim. Live-simulator closure also re-audits owner GPU proof
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
`rank_fidelity_result_proven` or `public_claim_upgrade_allowed` in `proof_boundary.json`.
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
not prove simulator execution or generated-world rank fidelity by itself.

Legacy evidence export and archive (deprecated compatibility command; not a SKU):

```bash
python -m blueprint_pipeline.post_training_data_package \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id>
```

The export also writes `rl_post_training_handoff_packet.json` into the evidence bundle
and archive so robot teams receive the same sparse reward, A/B reservation,
bottleneck, speed curriculum, action-chunk QA, and intervention/safety support
signals alongside curated traces and labels. Post-training use is permitted only
when the Task Evaluation Run's rights, provenance, robot-action alignment,
quality, and leakage gates pass. Export proves neither training, policy
improvement, deployment readiness, nor physical safety.

Visual augmentation support packet for optional evidence reuse and
distribution-shift review:

```bash
blueprint-build-oscar-visual-augmentation-packet \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --first-frame /path/to/first_frame.png \
  --skeleton-video /path/to/skeleton_conditioning.mp4 \
  --camera-provenance /path/to/camera_calibration_quality_gate.json \
  --skeleton-provenance /path/to/g1_projected_skeleton_trace.jsonl
```

Then run one generation job per variant through a visual-augmentation backend:

```bash
blueprint-run-oscar-visual-augmentation-generation \
  --packet-manifest /path/to/<job-dir>/oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json \
  --backend-id oscar_wam \
  --backend-mode auto
```

For local artifact/QA testing without a learned backend:

```bash
blueprint-run-oscar-visual-augmentation-generation \
  --packet-manifest /path/to/<job-dir>/oscar_visual_augmentation_packet/oscar_visual_augmentation_packet_manifest.json \
  --backend-mode fixture \
  --allow-fixture-backend
```

The real backend command contract is
`BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND` or
`BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND`. Existing
`BLUEPRINT_OSCAR_WAM_COMMAND`/`BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND` values use
the older WAM-rollout contract and should be wrapped before being used here.
The checked OSCAR Docker image is
`docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675023d4333767d798b565fc049ac5ba788cd7041db5cac7f9784fd49b3`
for the checked tag `20260701-cu128-ropefix`; it contains `/opt/oscar-public`
pinned to `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb` with the Blueprint
Torch-SDPA TransformerEngine compatibility shim. The checked linux/amd64
manifest digest is
`sha256:dc23334693d2983122f628ffaec9ea481bfdb8f0bfcec9d22efd83baba827b60`.

This prepares OSCAR/Cosmos/future-backend visual variant requests from fixed
camera/skeleton provenance and can attach reviewed backend outputs. Generated
videos from this packet are support assets only; they do not prove contact
physics, physical robot readiness, deployment approval, safety validation, or
real-world task success.
See [`docs/OSCAR_VISUAL_AUGMENTATION_PACKET.md`](docs/OSCAR_VISUAL_AUGMENTATION_PACKET.md).

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

The bridge contract for this repo is documented in [`docs/CAPTURE_BRIDGE_CONTRACT.md`](docs/CAPTURE_BRIDGE_CONTRACT.md).

Current cross-repo implementation status is tracked in [`docs/READINESS_MATRIX.md`](docs/READINESS_MATRIX.md). It is intentionally strict about what is shipped in-repo versus what still depends on live GPU/runtime/model access.
