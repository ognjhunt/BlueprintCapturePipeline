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
- `pipeline/cosmos3_readiness/cosmos3_capture_grounded_readiness.json`
- rights/privacy review artifacts when present

Optional actual evidence inputs may be staged under
`pipeline/robot_eval_inputs/`, but the current lane treats missing robot POV,
human demo, action logs, and actual outcomes as explicit missing-proof statuses.

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
  robot_task_library.json
  scenario_library.json
  robot_pov_evidence_requirements.json
  human_demo_evidence_requirements.json
  failure_taxonomy.json
  prediction_outcome_ledger.json
  eval_methodology_summary.md
```

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
- `robot_task_library_uri`
- `robot_scenario_library_uri`
- `robot_pov_evidence_requirements_uri`
- `human_demo_evidence_requirements_uri`
- `robot_failure_taxonomy_uri`
- `prediction_outcome_ledger_uri`
- `robot_eval_methodology_summary_uri`

## Fail-Closed Statuses

The dataset manifest uses these machine-readable statuses:

- `capture_grounded_ready`
- `needs_robot_pov`
- `needs_human_demo`
- `needs_action_logs`
- `needs_actual_outcome`
- `blocked_rights_privacy`
- `review_only_no_robot_readiness`

`capture_grounded_ready` means task/scenario records can be assembled from local
capture/package artifacts. It does not mean the robot is ready, the simulator
ran, or the site is operationally deployment-ready.

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
  physics/contact validation, robot policy execution, safety validation,
  external licensing, real pilot outcomes, and generated-scenario proof.

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
remain `needs_actual_outcome`.

## WebApp Boundary

`Blueprint-WebApp` may sync and display this lane only as:

- an advisory dataset contract
- task/scenario library summary
- evidence requirements
- failure taxonomy
- prediction-vs-actual ledger schema
- missing-proof labels

WebApp must not use these artifacts alone to claim:

- robot-ready or deployment-ready status
- safety validation
- simulator execution completed
- actual robot trial passed
- guaranteed success, cycle-time, or intervention thresholds

Those claims require owner-system proof from simulator traces, robot trials,
action logs, rights/privacy clearance, and buyer-approved methodology.

## Command

```bash
blueprint-build-robot-eval-dataset --capture-root /path/to/capture-root
```

The command is a local artifact writer only.
