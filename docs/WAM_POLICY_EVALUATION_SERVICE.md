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

## Live Or Owner WAM Adapter Path

`cosmos3_wam` and `oscar_wam` are adapter contracts, not hardwired product
dependencies. They fail closed unless all of these are present:

- an explicit local run gate: `--allow-wam-provider` for robot-eval jobs or
  `--allow-live-provider` for the WAM evaluator CLI
- `BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true`
- a provider command such as `--wam-provider-command cosmos3_wam=/path/to/adapter`
  or `BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND`
- provider auth in env only: Cosmos accepts one of
  `BLUEPRINT_COSMOS3_WAM_API_KEY`, `COSMOS_API_KEY`, or `NVIDIA_API_KEY`;
  OSCAR accepts one of `BLUEPRINT_OSCAR_WAM_API_KEY` or `OSCAR_WAM_API_KEY`

Example job-level adapter run:

```bash
BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true \
BLUEPRINT_COSMOS3_WAM_API_KEY=<redacted> \
blueprint-run-robot-eval-job \
  --capture-root /path/to/<capture-root> \
  --job-request /path/to/robot_eval_job_request.json \
  --job-id <job_id> \
  --provisioner fixture_local \
  --simulator fixture \
  --evaluation-substrate cosmos3_wam \
  --allow-wam-provider \
  --wam-provider-command cosmos3_wam="/path/to/cosmos_adapter" \
  --wam-artifact-output-uri gs://customer-bucket/<job_id>/wam \
  --wam-provider-max-retries 1 \
  --wam-provider-timeout-seconds 120
```

The adapter receives `BLUEPRINT_WAM_PROVIDER_INPUT`,
`BLUEPRINT_WAM_PROVIDER_OUTPUT`, `BLUEPRINT_WAM_PROVIDER_SUBSTRATE`, and,
when supplied, `BLUEPRINT_WAM_PROVIDER_ARTIFACT_OUTPUT_URI`. It must write JSON
with `rollouts` or `wam_rollout_results.rollouts`. Secrets must stay in env and
must not be written into artifacts.

## WAM Artifact Contract

When WAM evaluation is requested, the job writes:

- `evaluation_substrate_registry.json`
- `wam_evaluation_request.json`
- `wam_provider_runtime_package.json`
- `wam_provider_execution_manifest.json`
- `wam_provider_cost_control_ledger.json`
- `wam_provider_artifact_upload_proof.json`
- `wam_policy_interface_binding.json`
- `wam_rollout_manifest.json`
- `wam_rollout_results.json`
- `vision_success_labels.json`
- `wam_vision_success_review_queue.json`
- `normalized_attempt_trace.json`
- `failure_labels.json`
- `prediction_outcome_ledger.json`
- `calibration_report.json`
- `breakage_library.json`
- `policy_ranking_scorecard.json`
- `wam_eval_claim_boundary.json`
- `real_world_validation_followup_request.json`
- `srcc_validation_plan.json`
- `wam_real_world_validation_anchor_manifest.json`
- `wam_customer_validation_envelope.json`
- `wam_production_ops_manifest.json`
- `wam_classical_sim_cross_check_plan.json`
- `customer_handoff_report.json`
- `customer_handoff_report.md`

The fixture evaluator deterministically generates rollout support manifests,
fixture vision labels, normalized attempts, failure labels, a policy ranking
scorecard, a customer handoff report, and a real-world validation follow-up
request. Live providers are represented through the same artifacts and remain
blocked until adapter commands, auth envs, gates, and output rollouts are present.

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
