# WAM Policy Evaluation Service

## Purpose

Blueprint's robot-team service is capture-first and substrate-agnostic. A real
capture package produces Task Evaluation Run and Post-Training Data Package
artifacts; an evaluation substrate then generates support evidence for ranking
customer policies or checkpoints.

World-action-model evaluation is now first-class. Classical simulation remains
supported as a fallback, cross-check, or stricter physics lane.

## Local Substrates

The substrate registry is written as `evaluation_substrate_registry.json` and
currently supports:

- `fixture_wam`: deterministic repo-local WAM fixture for tests and local demos.
- `cosmos3_wam`: live or owner-provided Cosmos-style WAM adapter, blocked until
  configured.
- `oscar_wam`: live or owner-provided OSCAR-style WAM adapter, blocked until
  configured.
- `classical_sim_mujoco`: MuJoCo or owner-command simulator path.
- `classical_sim_isaac`: Isaac Sim / Isaac Lab / owner GPU simulator path.
- `recorded_trace`: customer or owner recorded trace replay path.

Legacy simulator aliases such as `mujoco`, `isaac_sim`, and `fixture` are
accepted at the contract boundary and normalized into the registry.

## Fixture End-to-End Path

The local fixture path requires no GPU, secrets, provider calls, or live VLM.
It starts from an existing robot-eval job directory with
`scenario_eval_matrix.json`, `policy_package_manifest.json`, and `job_request.json`.

```bash
blueprint-run-robot-eval-job \
  --capture-root /path/to/<capture-root> \
  --job-request /path/to/robot_eval_job_request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate fixture_wam
```

The WAM fixture can also be run directly against an existing job directory:

```bash
blueprint-run-wam-fixture-evaluator \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --evaluation-substrate fixture_wam
```

## WAM Artifact Contract

When WAM evaluation is requested, the job writes:

- `evaluation_substrate_registry.json`
- `wam_evaluation_request.json`
- `wam_rollout_manifest.json`
- `wam_rollout_results.json`
- `vision_success_labels.json`
- `normalized_attempt_trace.json`
- `failure_labels.json`
- `prediction_outcome_ledger.json`
- `calibration_report.json`
- `breakage_library.json`
- `policy_ranking_scorecard.json`
- `wam_eval_claim_boundary.json`
- `real_world_validation_followup_request.json`
- `srcc_validation_plan.json`
- `customer_handoff_report.json`
- `customer_handoff_report.md`

The fixture evaluator deterministically generates rollout support manifests,
fixture vision labels, normalized attempts, failure labels, a policy ranking
scorecard, a customer handoff report, and a real-world validation follow-up
request. Live providers are represented as blocked manifests until adapters and
explicit gates exist.

## Policy Autoresearch

Policy autoresearch can call a WAM evaluator command through the same command
hook used for MuJoCo:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --evaluation-substrate fixture_wam \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_wam_fixture_evaluator"
```

External evaluator commands receive both legacy and substrate-aware environment
variables:

- `BLUEPRINT_POLICY_AUTORESEARCH_SIMULATOR_ENGINE`
- `BLUEPRINT_POLICY_AUTORESEARCH_EVALUATION_SUBSTRATE`
- `BLUEPRINT_POLICY_AUTORESEARCH_MATRIX`
- `BLUEPRINT_POLICY_AUTORESEARCH_RECIPE`
- `BLUEPRINT_POLICY_AUTORESEARCH_OUTPUT`

Promotion still depends on the frozen scenario/eval matrix, train/heldout split,
success and failure labels, safety/contact gates, and claim boundaries.

## Claim Boundaries

Generated WAM rollouts are model-derived support artifacts. They are not raw
capture evidence, real robot rollouts, deployment approval, safety approval, or
public-readiness proof.

SC3-Eval-style and OSCAR/Cosmos-style results are credible research signals, but
they do not prove high SRCC for arbitrary customer hardware, policies, sites, or
task families. A customer-specific SRCC or Pearson claim requires paired real
world validation rollouts with exact `scenario_eval_run_id` joins, policy or
checkpoint IDs, and owner evidence or operator attestation.

Passing WAM heldout evaluation can support policy ranking, failure discovery,
and a real-world validation request. It cannot approve deployment by itself.
