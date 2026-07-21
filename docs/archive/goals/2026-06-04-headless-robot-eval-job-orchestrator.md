# Headless Robot-Eval Job Orchestrator Goal

> Archived historical goal record.

Date: 2026-06-04

Status: implemented locally with fixture-backed proof; real GPU provisioning,
real simulator execution, live Agents SDK orchestration, and Cosmos training
proof remain gated.

## Objective

Build the job-level orchestration layer that takes a robot-team request for a
policy/container/trace/demo/plugin, robot profile, site package, task/scenario
set, operation, simulator preference, training preference, budget,
rights/privacy scope, owner system, provenance, and timestamp alignment, then
drives the headless workflow until it completes or writes exact blocker
manifests.

The proof source remains deterministic manifests, traces, logs, and owner-system
artifacts. Agents may orchestrate, inspect, retry safe steps, summarize, and
route review, but they cannot become proof.

## Implemented Files

- `src/blueprint_pipeline/robot_eval_job_orchestrator.py`
- `src/blueprint_pipeline/evaluation_prep_stage.py`
- `tests/test_robot_eval_job_orchestrator.py`
- `pyproject.toml`
- `README.md`
- `docs/REAL_SITE_ROBOT_EVAL_DATASET_CONTRACT_2026-06-03.md`
- `docs/SIMULATION_AUTOMATION_LANE.md`
- `docs/architecture/command-safety-matrix.md`

## Local Outputs

`blueprint-run-robot-eval-job --capture-root <capture-root> --job-request
<request.json> --job-id <job_id>` writes:

- `pipeline/robot_eval_jobs/<job_id>/job_request.json`
- `pipeline/robot_eval_jobs/<job_id>/job_validation.json`
- `pipeline/robot_eval_jobs/<job_id>/job_plan.json`
- `pipeline/robot_eval_jobs/<job_id>/agent_orchestration_plan.json`
- `pipeline/robot_eval_jobs/<job_id>/gpu_provisioning_request.json`
- `pipeline/robot_eval_jobs/<job_id>/gpu_provisioning_result.json`
- `pipeline/robot_eval_jobs/<job_id>/simulator_service_request.json`
- `pipeline/robot_eval_jobs/<job_id>/simulator_service_result.json`
- `pipeline/robot_eval_jobs/<job_id>/policy_package_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/training_request.json`
- `pipeline/robot_eval_jobs/<job_id>/training_result.json`
- `pipeline/robot_eval_jobs/<job_id>/evaluation_request.json`
- `pipeline/robot_eval_jobs/<job_id>/evaluation_result.json`
- `pipeline/robot_eval_jobs/<job_id>/normalized_attempt_trace.json`
- `pipeline/robot_eval_jobs/<job_id>/failure_labels.json`
- `pipeline/robot_eval_jobs/<job_id>/prediction_outcome_ledger.json`
- `pipeline/robot_eval_jobs/<job_id>/calibration_report.json`
- `pipeline/robot_eval_jobs/<job_id>/breakage_library.json`
- `pipeline/robot_eval_jobs/<job_id>/proof_boundary.json`
- `pipeline/robot_eval_jobs/<job_id>/job_run_manifest.json`
- `pipeline/robot_eval_jobs/<job_id>/blocked_manifest.json` when blocked

## Checklist

- [x] Add repo-local goal/architecture doc.
- [x] Add robot eval job request and policy package validation contract.
- [x] Add deterministic job orchestrator module and CLI.
- [x] Write per-job state machine artifacts under `pipeline/robot_eval_jobs/<job_id>/`.
- [x] Add fake/local agent orchestrator and gated live Agents SDK operator adapter.
- [x] Add fixture provisioner and fail-closed real provider request/result manifests.
- [x] Add fixture simulator path and gated command simulator path for MuJoCo,
      PyBullet, Newton, and Isaac Sim.
- [x] Add export-only training request plus gated command training result path.
- [x] Validate policy API endpoint, Docker container, recorded action trace,
      high-level skill trace, teleop demo, and sim controller plugin references.
- [x] Add `blueprint-run-robot-eval-job` entrypoint.
- [x] Surface robot-eval job artifacts through evaluation prep as advisory URIs.
- [x] Update README, simulation lane, command safety matrix, and dataset contract.
- [x] Test request parsing, rights/privacy blockers, missing policy evidence,
      fixture success/failure, generated scenario review gating, command
      simulator gates, command output capture, training gates, agent adapters,
      no public claim upgrade, and evaluation-prep surfacing.

## Proof Boundary

Fixture jobs prove only that the repo-local orchestration state machine can
validate requests, allocate fixture provisioning, invoke the fixture simulator
path, copy Site Eval Director normalization/calibration/breakage artifacts, and
write deterministic job manifests.

They do not prove:

- Vast, RunPod, GCP, local process, or Docker provisioning
- MuJoCo, PyBullet, Newton, or Isaac Sim execution
- robot policy execution
- real robot trial success
- physics/contact validity
- off-scope validation
- Cosmos generation or training completion
- public generated-world rank fidelity

Those upgrades require explicit environment and CLI gates plus owner-system
result manifests, checkpoint artifacts where relevant, rights/privacy clearance,
accepted methodology, and human/buyer approval where applicable.

## Verification

Focused local proof:

```bash
PYTHONPATH=src pytest tests/test_robot_eval_dataset.py tests/test_site_eval_director.py tests/test_simulation_automation.py tests/test_robot_eval_job_orchestrator.py -q
```

CLI help proof:

```bash
PYTHONPATH=src python -m blueprint_pipeline.robot_eval_job_orchestrator --help
PYTHONPATH=src python -m blueprint_pipeline.site_eval_director --help
PYTHONPATH=src python -m blueprint_pipeline.simulation_automation --help
PYTHONPATH=src python -m blueprint_pipeline.robot_eval_dataset --help
```
