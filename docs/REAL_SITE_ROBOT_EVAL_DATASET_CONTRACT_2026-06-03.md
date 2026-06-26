# Real-Site Robot Eval Dataset Contract

Date: 2026-06-03

Status: repo-local deterministic artifact contract.

Owner repo: `BlueprintCapturePipeline`

## Purpose

This lane defines the first real-site robot evaluation dataset and workflow
contract for Blueprint site packages. It turns capture-grounded package evidence
into versioned Site, Task, Scenario, and Eval Card artifacts plus robot POV and
human-demo evidence requirements, failure labels, annotation backlog, proof
boundaries, and prediction-vs-actual outcome records.

It does not call live providers, run simulators, download models, send messages,
touch payments, deploy, or upgrade any public claim.

## Inputs

The writer reads local artifacts only:

- `capture_descriptor.json`
- `raw/manifest.json`
- `pipeline/evaluation_prep/task_anchor_manifest.json`
- `pipeline/evaluation_prep/object_geometry_manifest.json`
- `pipeline/evaluation_prep/site_world_spec.json`
- `pipeline/evaluation_prep/hosted_session_runtime_manifest.json`
- `pipeline/simready/simready_scene_manifest.json`
- `pipeline/simready/simready_validation.json`
- `pipeline/marble_sim_assets/marble_simready_bridge.json`
- `pipeline/marble_sim_assets/marble_asset_validation.json`
- `pipeline/simulation_automation/scene_asset_inventory.json`
- `pipeline/simulation_automation/scene_asset_dependency_audit.json`
- `pipeline/simulation_automation/scene_asset_preflight.json`
- `pipeline/simulation_automation/scene_asset_inspection.json`
- `pipeline/simulation_automation/scene_frame_estimate.json`
- `pipeline/simulation_automation/collider_proxy_plan.json`
- `pipeline/simulation_automation/cpu_scene_proxy_manifest.json`
- `pipeline/simulation_automation/cpu_preflight_scorecard.json`
- `pipeline/simulation_automation/task_anchor_proposal_manifest.json`
- `pipeline/simulation_automation/episode_spec_manifest.json`
- `pipeline/simulation_automation/episode_specs.json`
- `pipeline/simulation_automation/spawn_pose_validation_manifest.json`
- `pipeline/simulation_automation/cpu_preflight_manifest.json`
- `pipeline/simulation_automation/pre_gpu_readiness_summary.json`
- `pipeline/simulation_automation/cpu_simulator_preflight_manifest.json`
- `pipeline/simulation_automation/gpu_handoff_packet.json`
- `pipeline/simulation_automation/gpu_owner_system_proof_schema.json`
- `pipeline/simulation_automation/gpu_run_checklist.md`
- `pipeline/simulation_automation/owner_gpu_simulator_execution_blocked_manifest.json`
- `pipeline/cosmos3_readiness/cosmos3_capture_grounded_readiness.json`
- rights/privacy review artifacts when present

Optional actual evidence inputs may be staged under
`pipeline/robot_eval_inputs/`, but the current lane treats missing robot POV,
human demo, action logs, and actual outcomes as explicit missing-proof statuses.

Robot-team submission references may also be staged as:

- `pipeline/robot_eval_inputs/robot_team_test_submission_manifest.json`

That optional input mirrors the WebApp hosted-session policy field
`policy.robotTeamTestSubmission` and uses schema version
`blueprint.robot_team_test_submission.v1`.

Robot-team headless evaluation or training jobs may be staged as local request
manifests and run through `blueprint-run-robot-eval-job`. The job request
represents customer, site/package, task/scenario, robot profile, the six
robot-team submission modalities, requested operation, simulator preference,
Cosmos/training preference, budget/time limits, rights/privacy scope, owner
system, provenance, and timestamp alignment. The job layer validates and
orchestrates around this dataset contract; it does not replace Site, Task,
Scenario, or Eval Cards as the deterministic evidence surface.

WebApp-created `robot_eval_job_request.v1` envelopes carry a simulator routing
policy instead of selecting a live runtime by themselves. The current policy is
`mujoco_first_unless_proof_requires_isaac`: Pipeline should recommend MuJoCo for
the first cheap real simulator pass unless the request names richer USD/OpenUSD,
Isaac robot-asset, RTX sensor, contact/physics, or batch Arena proof classes.
Isaac Sim and Isaac Lab/Arena remain escalation backends for those proof classes.
MuJoCo proof does not clear Isaac-specific gates, real robot POV, safety/contact,
delivery, or public-claim upgrades.

## Outputs

The writer emits:

```text
pipeline/robot_eval_dataset/
  robot_eval_dataset_manifest.json
  real_site_robot_eval_dataset_manifest.json
  site_card.json
  task_cards.json
  scenario_cards.json
  eval_cards.json
  annotation_backlog.json
  proof_boundaries.json
  rights_packet.json
  rights_ledger.json
  robot_task_library.json
  task_ontology_v1.json
  scenario_library.json
  scenario_family_library.json
  robot_pov_evidence_requirements.json
  human_demo_evidence_requirements.json
  robot_eval_inputs_evidence_contract.json
  robot_team_test_submission_modalities.json
  failure_taxonomy.json
  scoring_methodology.json
  task_thresholds.json
  publication_readiness.json
  recorded_trace_eval_report.json
  policy_eval_report.json
  prediction_outcome_ledger.json
  prediction_vs_actual_summary.json
  eval_methodology_summary.md

pipeline/simulation_automation/ (advisory references only)
  scene_asset_inventory.json
  scene_asset_dependency_audit.json
  scene_asset_preflight.json
  collider_proxy_plan.json
  cpu_scene_proxy_manifest.json
  cpu_preflight_scorecard.json
  task_anchor_proposal_manifest.json
  episode_spec_manifest.json
  episode_specs.json
  spawn_pose_validation_manifest.json
  cpu_preflight_manifest.json
  pre_gpu_readiness_summary.json
  cpu_simulator_preflight_manifest.json
  gpu_handoff_packet.json
  gpu_owner_system_proof_schema.json
  gpu_run_checklist.md
  owner_gpu_simulator_execution_blocked_manifest.json

pipeline/robot_eval_jobs/<job_id>/
  job_request.json
  job_validation.json
  job_plan.json
  agent_orchestration_plan.json
  scheduler_decision.json
  worker_launch_plan.json
  worker_manifest.json
  gpu_provisioning_request.json
  gpu_provider_launch_request.json
  gpu_cost_control_ledger.json
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
  startup_architecture_audit.json
  job_run_manifest.json
  blocked_manifest.json

pipeline/robot_eval_job_requests/
  <job_id>/job_request.json
  inbox_run_manifest.json
```

`publication_readiness.json` is the pre-publication gate for WebApp. A site may
be shown as `Ready to evaluate` only when the required Site/Task/Scenario/Eval
Card family, proof boundaries, task ontology, scenario family library, scoring
methodology, `task_thresholds.json`, and `publication_readiness.json` are
present. Missing robot POV, action logs, actual outcomes, policy references, or
owner-system proof stay as missing-proof labels and must not become readiness
claims.

Evaluation prep includes these paths in
`pipeline/evaluation_prep/evaluation_prep_manifest.json.artifacts` and exposes
URI fields for WebApp sync:

- `robot_eval_dataset_manifest_uri`
- `robot_eval_legacy_manifest_uri`
- `robot_eval_site_card_uri`
- `robot_eval_task_cards_uri`
- `robot_eval_scenario_cards_uri`
- `robot_eval_cards_uri`
- `robot_eval_annotation_backlog_uri`
- `robot_eval_proof_boundaries_uri`
- `robot_rights_packet_uri`
- `robot_rights_ledger_uri`
- `robot_task_library_uri`
- `robot_task_ontology_v1_uri`
- `robot_scenario_library_uri`
- `robot_scenario_family_library_uri`
- `robot_pov_evidence_requirements_uri`
- `human_demo_evidence_requirements_uri`
- `robot_eval_inputs_evidence_contract_uri`
- `robot_team_test_submission_modalities_uri`
- `robot_failure_taxonomy_uri`
- `robot_scoring_methodology_uri`
- `robot_eval_task_thresholds_uri`
- `robot_eval_publication_readiness_uri`
- `recorded_trace_eval_report_uri`
- `policy_eval_report_uri`
- `prediction_outcome_ledger_uri`
- `prediction_vs_actual_summary_uri`
- `robot_eval_methodology_summary_uri`
- `robot_eval_job_<job_id>_run_manifest_uri`
- `robot_eval_job_<job_id>_proof_boundary_uri`
- `robot_eval_job_<job_id>_blocked_manifest_uri` when blocked
- stable latest-job aliases:
  `robot_eval_job_request_uri`, `robot_eval_job_run_manifest_uri`,
  `robot_eval_job_proof_boundary_uri`, and
  `robot_eval_job_blocked_manifest_uri`

## Fail-Closed Statuses

The dataset manifest uses these machine-readable statuses:

- `capture_grounded_ready`
- `needs_robot_pov`
- `needs_human_demo`
- `needs_action_logs`
- `needs_actual_outcome`
- `needs_policy_api_endpoint_ref`
- `needs_docker_container_ref`
- `needs_recorded_action_trace_ref`
- `needs_high_level_skill_trace_ref`
- `needs_teleop_demo_ref`
- `needs_sim_controller_plugin_ref`
- `blocked_rights_privacy`
- `review_only_no_rank_fidelity`

`capture_grounded_ready` means task/scenario records can be assembled from local
capture/package artifacts. It does not mean the robot is ready, the simulator
ran, or the site is operationally rank-fidelity-scored.

## v0.1 Card Families

- `site_card.json`: site type, geometry, visual conditions, dynamic conditions,
  safety constraints, robot metadata, provenance, rights/privacy, and review
  status for one capture-backed site. Missing collider evidence keeps collision
  and physics/contact readiness claims blocked.
- `task_cards.json`: task statements, start state, success/failure definitions,
  required metrics, task evidence source, confidence, and missing annotations.
- `scenario_cards.json`: normal scenario, variation, edge case, known risk,
  observed-vs-inferred labels, and required missing annotations. Generated or
  inferred variations are never real-world proof.
- `eval_cards.json`: robot/policy scope, engine used, prediction placeholders,
  failure modes, intervention estimate, uncertainty, validation status, proof
  boundary, and blocked upgrades.
- `annotation_backlog.json`: missing robot POV, human demo, action logs, actual
  outcomes, rights/privacy, collider, and operator-review annotations.
- `proof_boundaries.json`: fail-closed booleans for simulator execution,
  physics/contact validation, robot policy execution, off-scope validation,
  external licensing, real pilot outcomes, and generated-scenario proof.
- `rights_packet.json` and `rights_ledger.json`: raw confidential data,
  derived/de-identified environment, synthetic variant, robot-eval, commercial
  licensing, revenue-share, exclusivity, expiration, approver, evidence URI,
  blocker, allowed-use, and disallowed-use review records. These records are
  packets and ledgers only; they do not clear public or commercial use by
  themselves.
- `task_ontology_v1.json`: canonical cross-site task IDs, aliases, parameters,
  success criteria, evidence requirements, supported metrics, and query fields.
- `scenario_family_library.json`: capture-grounded, representative-mock,
  agent-inferred-needs-review, accepted, rejected, and review-only scenario
  family records for lighting variation, object rotation, cart shifted, blocked
  path, human crossing, forklift nearby, occlusion, glare, missing label, wrong
  object nearby, and narrow approach angle. It never treats generated scenarios
  as simulator or real-world proof.
- `robot_eval_jobs/<job_id>/scenario_eval_matrix.json`: job-level expansion from
  requested task/scenario scope to concrete scenario-family variation runs. Robot
  POV observations, policy adapters, simulator adapters, closure coverage checks,
  and package exports use this matrix so every required variation family stays
  visible through evaluation.
- `robot_eval_jobs/<job_id>/failure_labels.json`: failed-attempt label coverage.
  Every failed attempt or failed `scenario_eval_run_id` from
  `normalized_attempt_trace.json` must have a corresponding failure label before
  the live closure failure-label gate can pass.
- `scoring_methodology.json`: versioned deterministic scoring methodology for
  success rate, cycle time, intervention rate, unsafe proximity, collision risk,
  object drop, wrong object, timeout, recovery success, uncertainty, and a
  sim-vs-real calibration placeholder. The job-level standard scorecard must use
  valid numeric ranges and object shapes before the evaluation-methodology
  closure gate can pass; field presence alone is not sufficient.
- `recorded_trace_eval_report.json` / `policy_eval_report.json`: local
  recorded-action-trace fixture runner output. It scores recorded traces without
  Docker, policy API, simulator, live provider, or credential requirements, and
  remains advisory.
- `prediction_vs_actual_summary.json`: deterministic ingestion summary for
  pilot, teleop, operator, or recorded-trace outcome manifests. Missing actuals
  remain blocked/advisory.
- `robot_team_test_submission_modalities.json`: schema version
  `robot_team_test_submission_modalities.v0.1`, the six accepted WebApp
  submission modalities, their required camelCase policy field keys, missing
  evidence statuses, accepted artifact-reference policy, and blocked claim
  upgrades.

## Structured Robot-Team Submission Modalities

The Pipeline modality artifact is the schema source for the WebApp structured
test interface. It currently names these modalities:

- `policy_api_endpoint`
- `docker_container`
- `recorded_action_trace`
- `high_level_skill_trace`
- `teleop_demo`
- `sim_controller_plugin`

The required reference fields intentionally match the WebApp
`policy.robotTeamTestSubmission` payload keys. Missing references are tracked as
fail-closed statuses; present references only move a modality to
`reference_present_requires_owner_system_review`. They do not prove a policy
ran, a simulator completed, a real robot trial passed, or a deployment threshold
was met.

The job orchestrator validates the same six modalities before provisioning or
execution. Missing or weak references write exact `needs_*_ref` statuses into
`policy_package_manifest.json`, `job_validation.json`, and
`blocked_manifest.json`. The live closure policy-interface gate revalidates the
selected modalities and blocks any selected modality that is unsupported,
`not_selected`, `blocked`, or missing its modality-specific required reference
fields.

## Ledger Methodology

`prediction_outcome_ledger.json` supports these prediction sources:

- `marble_review`
- `simready_review`
- `cosmos_preflight`
- `human_eval`
- `future_provider`
- `simulator_trace`
- `robot_trial`

It supports these actual sources:

- `heldout_revisit`
- `robot_pilot`
- `simulator_trace`
- `human_demo`
- `teleop`
- `operator_report`

Records carry success/failure, task completion, cycle time, intervention count,
contact/collision events, safety violations, failure mode IDs, confidence, proof
artifact paths, owner system, and claim boundary fields. Missing actual outcomes
remain `needs_actual_outcome`. Actual outcome records can be tracked and used for
sim-vs-real calibration when they pair with predictions. Records that include a
`scenario_eval_run_id` must pair with the prediction for that same run; unmatched
records remain calibration blockers and are surfaced in the deployment ledger and
live closure manifest. Actual outcomes do not satisfy `real_world_outcome_proven`
unless every actual outcome row includes owner evidence refs, an owner proof URI,
or an operator/owner attestation.

## WebApp Boundary

`Blueprint-WebApp` may sync and display this lane only as:

- an advisory dataset contract
- task/scenario library summary
- evidence requirements
- robot-team submission modality requirements
- failure taxonomy
- prediction-vs-actual ledger schema
- missing-proof labels
- CPU preflight scorecard status
- episode spec summary
- CPU simulator preflight status
- backend-specific collider blockers such as `isaac_usd_collision_unverified`,
  `portable_collider_glb_missing`, `cpu_proxy_collision_estimated`, and
  `simulator_execution_not_run`

WebApp must not use these artifacts alone to claim:

- robot-ready or rank-fidelity-scored status
- off-scope validation
- simulator execution completed
- local CPU preflight smoke as accepted simulator execution
- actual robot trial passed
- submitted policy/container/trace/demo/plugin passed evaluation
- headless robot-eval job success as real simulator, real training, or real
  robot proof
- guaranteed success, cycle-time, or intervention thresholds

Those claims require owner-system proof from simulator traces, robot trials,
action logs, rights/privacy clearance, and buyer-approved methodology.

## Command

```bash
blueprint-build-robot-eval-dataset --capture-root /path/to/capture-root
```

The command is a local artifact writer only.
