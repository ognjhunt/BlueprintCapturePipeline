# Headless Eval Director Loop Goal

> Archived historical goal record.

Date: 2026-06-04

Status: implemented locally with fixture-backed proof; real simulator, real robot, live provider, and Cosmos training proof remain gated.

## Objective

Implement the post-Marble/World Labs headless robot evaluation loop while keeping Marble, 3DGS/mesh, Isaac/MuJoCo/PyBullet/Newton, Cosmos, and pilot evidence replaceable behind stable Site, Task, Scenario, and Eval Cards.

The platform-owned loop is:

1. site capture and task definitions
2. robot POV, human demo, action log, simulator trace, policy submission, and actual outcome evidence contracts
3. deterministic fixture-backed runner abstraction
4. normalized scenario attempt traces
5. success/failure label application against `failure_taxonomy.json`
6. predicted-vs-actual calibration
7. learned facility-breakage records
8. Cosmos export/request manifests only
9. proof-boundary manifests that do not upgrade public claims

## Implemented Files

- `src/blueprint_pipeline/robot_eval_dataset.py`
- `src/blueprint_pipeline/site_eval_director.py`
- `src/blueprint_pipeline/evaluation_prep_stage.py`
- `tests/test_robot_eval_dataset.py`
- `tests/test_site_eval_director.py`
- `README.md`
- `docs/REAL_SITE_ROBOT_EVAL_DATASET_CONTRACT_2026-06-03.md`
- `docs/SIMULATION_AUTOMATION_LANE.md`
- `docs/architecture/command-safety-matrix.md`

## Local Outputs

`blueprint-run-site-eval-director --capture-root <capture-root>` writes:

- `pipeline/simulation_automation/scenario_execution_plan.json`
- `pipeline/simulation_automation/task_simulation_requests.json`
- `pipeline/simulation_automation/scenario_simulator_matrix.json`
- `pipeline/simulation_automation/normalized_simulator_attempt_trace.json`
- `pipeline/simulation_automation/failure_labels.json`
- `pipeline/simulation_automation/updated_eval_cards.json`
- `pipeline/simulation_automation/site_eval_prediction_outcome_ledger.json`
- `pipeline/simulation_automation/site_eval_calibration_report.json`
- `pipeline/simulation_automation/learned_facility_breakage_library.json`
- `pipeline/simulation_automation/cosmos_orchestration_exports.json`
- `pipeline/simulation_automation/site_eval_director_proof_boundary.json`
- `pipeline/simulation_automation/site_eval_director_run_manifest.json`
- `pipeline/simulation_automation/site_eval_real_evidence_blocked_manifest.json` when real owner-system evidence is absent
- `pipeline/simulation_automation/site_eval_fixture_runner_blocked_manifest.json` when fixture execution is blocked

## Done When Checklist

- [x] Add repo-local goal/architecture doc under `docs/goals/2026-06-04-headless-eval-director-loop.md`.
- [x] Add evidence ingestion contracts for `pipeline/robot_eval_inputs`.
- [x] Add simulator runner abstraction with fixture runner first and real engines fail-closed.
- [x] Upgrade Site Eval Director into a deterministic fixture-backed local loop.
- [x] Keep real simulator execution fail-closed unless env and CLI gates are present.
- [x] Implement success/failure label application with automatic, human-reviewed, and review-required status support.
- [x] Implement predicted-vs-actual calibration deltas and aggregation by site/task/scenario/policy/engine.
- [x] Implement learned facility-breakage aggregation.
- [x] Keep Cosmos outputs as export/request manifests unless explicit proof exists.
- [x] Wire CLI, docs, and evaluation-prep artifact surfaces.
- [x] Test blocked path, fixture success, fixture failure, label mapping, calibration deltas, breakage aggregation, rights/privacy blocking, generated scenario review gating, real-engine fail-closed behavior, and no public claim upgrade.

## Proof Boundary

Fixture attempts prove only that the repo-local headless loop can select scenarios, normalize attempts, label outcomes, calibrate predictions, aggregate breakage, and write deterministic artifacts.

They do not prove:

- Isaac/MuJoCo/PyBullet/Newton execution
- physics/contact validity
- robot policy execution
- real robot trial success
- off-scope validation
- Cosmos generation or training completion
- public generated-world rank fidelity

Those upgrades require owner-system simulator traces or pilot logs, rights/privacy clearance, accepted methodology, and explicit execution gates.
