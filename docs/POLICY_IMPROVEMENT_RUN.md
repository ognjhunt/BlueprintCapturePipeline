# Policy Improvement Run

## Objective

Policy Improvement Run is Blueprint's managed policy-lift offer for robot teams.
It turns a real-site Task Evaluation Run and Post-Training Data Package into a
bounded attempt to improve a customer-supplied policy, adapter, task head,
distilled skill, or complete policy candidate. WAM/substrate evaluation is now
first-class; classical simulation remains available as a fallback, cross-check,
or stricter physics lane.

The offer is more valuable than evaluation alone when a team already has a
borderline policy and needs evidence that recoverable site/task failures can be
closed before a pilot. The sales promise is not "Blueprint owns the robot
foundation model." The sales promise is:

> Turn a failed or borderline site eval into a better policy candidate with
> auditable before/after evidence.

## Customer Inputs

- Policy or base model: API, container, action trace, adapter checkpoint, task
  head, training recipe, or source repo.
- Robot embodiment: robot model, kinematic limits, sensors, payload limits, and
  safety envelope.
- Action interface: command schema, frequency, units, limits, and reset
  behavior.
- Target task: task statement, operating envelope, fixtures, objects, and
  allowed recovery behavior.
- Required success threshold: for example `0.95`.
- Required cycle-time threshold: for example `90` seconds.

Source code is optional by default. The access ladder is:

- `black_box`: policy API/container/action traces. Blueprint can evaluate,
  diagnose failures, tune wrappers, create curricula, and produce evidence.
- `config_adapter`: adapter, task-head, or policy configuration access.
  Blueprint can post-train a bounded adapter, task head, or distilled skill.
- `source_training`: source and training access. Blueprint can attempt broader
  training recipe or complete policy changes, but sealed scoring remains
  immutable.

## Workflow

1. Build or reuse the real-site Task Evaluation Run.
2. Select an evaluation substrate such as `fixture_wam`, `cosmos3_wam`,
   `oscar_wam`, `classical_sim_mujoco`, `classical_sim_isaac`, or
   `recorded_trace`.
3. Evaluate the baseline policy.
4. Identify dominant failure modes from normalized attempts, clips, labels, and
   review ledgers.
5. Generate twin and cousin scenarios from the site/task distribution.
6. Build a curriculum with development, validation, heldout, and sealed audit
   splits.
7. Post-train or lift a candidate artifact.
8. Test candidate versions on heldout or sealed scenarios that were not used as
   training data.
9. Deliver the improved artifact, WebApp-safe summary projection, and evidence report.

The run now mirrors the Task Evaluation Run shape instead of being only a thin
offer wrapper: it carries a baseline evaluation scorecard projection, a staged
readiness ladder, and a `policy_improvement_run_webapp_summary.json` document
that WebApp can store without dense traces, secrets, training payloads, or raw
policy artifacts.

## Scenario Split Contract

- Development scenarios may be visible to the autoresearch loop and used for
  training.
- Validation scenarios may provide limited feedback for candidate selection.
- Heldout scenarios are required for promotion.
- Sealed audit scenarios must be inaccessible to training and cannot be changed
  by the research agent.

If the same 500 scenarios are repeatedly observed and used to improve the
policy, they have become training data. That score is no longer an independent
deployment-readiness measure.

## Commands

Build a Policy Improvement Run offer manifest:

```bash
blueprint-build-policy-improvement-run \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --access-level config_adapter \
  --customer-policy-ref customer-policy-v3 \
  --embodiment g1-humanoid \
  --action-interface joint_position_delta_20hz \
  --target-task tote-transfer \
  --success-threshold 0.95 \
  --cycle-time-threshold-seconds 90 \
  --improvement-target adapter \
  --improvement-target task_head
```

Run a WAM/substrate policy-autoresearch loop:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --evaluation-substrate fixture_wam \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_wam_fixture_evaluator"
```

For live or owner-provided WAM adapters, create the robot-eval job with
`--evaluation-substrate cosmos3_wam` or `--evaluation-substrate oscar_wam`,
`--allow-wam-provider`, the matching `--wam-provider-command`, and env-only
provider auth plus `BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER=true` first.
Autoresearch can then call substrate-specific evaluator commands with
`--evaluator-command-by-engine cosmos3_wam="..."` or
`--evaluator-command-by-engine oscar_wam="..."`; those commands still write only
support evidence and do not turn a WAM heldout pass into deployment approval.

Run the MuJoCo cross-check/fallback policy-autoresearch loop:

```bash
blueprint-run-policy-autoresearch \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id> \
  --policy-recipe /path/to/seed_policy_recipe.json \
  --reviewed-examples /path/to/reviewed_success_failure_examples.json \
  --simulator-engine mujoco \
  --evaluator-command "python -m blueprint_pipeline.policy_autoresearch_mujoco_evaluator"
```

Build the Post-Training Data Package:

```bash
blueprint-build-post-training-data-package \
  --capture-root /path/to/<capture-root> \
  --job-dir /path/to/<capture-root>/pipeline/robot_eval_jobs/<job_id>
```

Run focused tests:

```bash
python -m pytest tests/test_policy_improvement_run.py tests/test_policy_autoresearch.py
```

## Project Structure

- `src/blueprint_pipeline/policy_improvement_run.py`: offer manifest builder.
- `src/blueprint_pipeline/policy_autoresearch.py`: substrate-aware candidate
  search and promotion loop.
- `src/blueprint_pipeline/wam_fixture_evaluator.py`: deterministic local WAM
  job evaluator.
- `src/blueprint_pipeline/policy_autoresearch_wam_fixture_evaluator.py`:
  deterministic local WAM split evaluator for policy autoresearch.
- `src/blueprint_pipeline/post_training_data_package.py`: package export,
  checksums, optional exports, and archive.
- `tests/test_policy_improvement_run.py`: product contract coverage.
- `docs/POLICY_IMPROVEMENT_RUN.md`: this spec.

## Boundaries

- Always preserve raw capture, rights, privacy, and provenance truth.
- Always keep model backends replaceable and customer policies supported through
  API/container/action-trace paths.
- Always separate development, validation, heldout, and sealed audit scenario
  pools.
- Never let the policy-improvement agent mutate final scoring, success
  thresholds, sealed seeds, or proof boundaries.
- Never claim simulator heldout success is physical deployment approval.
- Never claim WAM heldout success, generated rollout labels, or a policy
  ranking scorecard is real-world deployment approval.
- Never claim customer-specific SRCC without paired real-world validation
  rollouts for that customer's hardware/policy/task family.
- Never require source access for the baseline commercial offer.

## Success Criteria

- The repo has a named Policy Improvement Run artifact contract.
- The offer clearly extends Task Evaluation Runs and Post-Training Data
  Packages.
- Customer-supplied policies work without source code access.
- The artifact reports baseline score, improved candidate score, failure modes,
  access level, readiness ladder, WebApp-safe projection, and proof limits.
- WAM outputs, when present, are labeled as model-derived support artifacts and
  include a real-world validation follow-up request.
- Missing customer inputs, missing heldout/sealed splits, or missing candidate
  evidence fail closed.

## Open Questions

- Which buyer-facing name should ship first: Policy Improvement Run, Policy
  Lift Sprint, or Pilot Gate Improvement Run?
- Which tasks should be the first paid wedge: navigation, tote transfer, mobile
  manipulation, or inspection?
- When should Blueprint accept responsibility for delivering complete policy
  weights rather than adapters or task heads?
