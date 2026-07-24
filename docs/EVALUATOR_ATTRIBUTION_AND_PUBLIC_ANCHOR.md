# Evaluator attribution, small-sample honesty, and the public anchor

This document covers a group of changes that share one premise: Blueprint's
evaluator gates were fail-closed in the right places but were resting on
statistics that could not support them, and on an anchor path that could never
be reached.

Nothing here measures a robot, runs a world model, or upgrades a claim. Every
artifact remains generated-media or third-party review evidence.

## 1. Why the policy cohort is the binding constraint

Public rank-fidelity claims are gated on a confidence-interval **lower bound**
(`robot_eval_calibration.py`, `sc3_fidelity_contracts.py`), which is the correct
posture. But a correlation's degrees of freedom come from the number of
independent policies, not the number of rollouts. Backing each policy with more
trials tightens each policy's own estimate; it does not add a point to the
correlation.

95% Fisher-z lower bound on Pearson *r* by policy count:

| point estimate *r* | n=7 | n=8 | n=12 | n=20 | n=33 |
| --- | --- | --- | --- | --- | --- |
| 0.929 | 0.586 | 0.650 | 0.761 | 0.826 | 0.860 |
| 0.950 | 0.692 | 0.742 | 0.827 | 0.876 | 0.900 |
| 0.989 | 0.924 | 0.938 | 0.960 | 0.972 | 0.978 |

At the seven-policy minimum, a realistic *r* = 0.95 certifies only ≥ 0.69.
Certifying ≥ 0.90 from a 0.95 estimate needs roughly **33 policies**. The
published headlines Blueprint tracks sit in exactly this regime: RoboWorld's
r = 0.989 rests on 8 policies (with 4,186 rollouts behind them), and SC3-Eval's
0.929 rests on 7 checkpoints.

`blueprint_pipeline.rank_fidelity_statistics` makes this computable rather than
implicit: Fisher-z intervals defined at every n ≥ 4, Wilson intervals for
proportions, the two-proportion minimum-detectable-difference curve, an exact
one-sided Fisher test, and a bootstrap-reliability judgement.

## 2. Pearson is demoted; pairwise ordering is the headline

`build_external_rank_fidelity_report` now emits a `headline` block naming
`pairwise_ordering_accuracy` as the reported metric, with `metric_roles`
marking Pearson as `supporting_fragile_at_small_cohorts`. Pairwise ordering is
what a buyer's question actually reduces to, it degrades gracefully at small
cohorts, and its Wilson interval stays inside `[0, 1]`.

The report also carries a `resolving_power` curve — the smallest success-rate
gap a design can detect at a given per-arm trial count — because the commercial
question is usually "is my 80k checkpoint better than my 60k checkpoint", where
the honest answer is often *indistinguishable at this trial count*.

## 3. The bootstrap no longer hides its own degeneracy

The percentile bootstrap resamples policies, and a resample that is constant in
either coordinate leaves Pearson undefined. Those replicates were discarded
silently, which **narrows** the published interval rather than widening it.

Each metric now reports `bootstrap_replicates_attempted`,
`bootstrap_replicates_defined`, an `undefined_replicate_fraction`, and
`confidence_interval_95_reliable`. An interval computed from a cohort below
`MIN_RELIABLE_BOOTSTRAP_SAMPLE_COUNT`, or one that dropped more than 5% of its
replicates, is marked unreliable instead of being presented as tight.

## 4. The policy ladder's acceptance statistic

The ladder accepted `recovered` when per-rung empirical success rates formed a
strict descending order. The builder hardcoded three replicate seeds, so the
attainable rates were 0, ⅓, ⅔ and 1 — adjacent rungs differed by a single
success, and the **exact one-sided p-value for that difference is 0.5**. The
ordering was as likely to have arisen by chance as not.

Changes:

- `_ladder_separation_analysis` computes an exact one-sided Fisher p-value for
  every adjacent rung pair, **before** the pass/fail decision, and reports it
  even when the run is otherwise blocked.
- `ladder_empirical_separation_not_statistically_resolvable` blocks acceptance
  when any adjacent pair is unresolvable, with the dedicated status
  `inconclusive_underpowered_separation`.
- `replicate_seed_count` is now a builder parameter. Its default
  (`DEFAULT_LADDER_SEED_COUNT = 63`) is derived from the separation the ladder
  is built to resolve rather than guessed, and `MIN_LADDER_SEED_COUNT` is
  documented as a structural floor that carries no statistical meaning.
- The report states how many seeds the observed separation would actually need.

## 5. A world-model-free control arm

Every published evaluator result Blueprint tracks ablates its method against
other configurations of itself. None report what a ranker with **no world
model** achieves on the same cohort — so the headline number is causally
unattributed.

`blueprint_pipeline.control_ranker` supplies the control arm. Baselines read
only what exists before any generation: action-chunk jerk, gripper toggle rate,
episode timeout rate, a first-frame-only prior, plus `constant` and
`seeded_pseudo_random` null controls reported separately.

The report's `attribution` block gives the evaluator's marginal contribution
over the best world-model-free baseline, with a **paired** bootstrap over
policies — resampling the cohort once per replicate and recomputing both arms on
the same resample — rather than subtracting two independently-wide point
estimates.

A baseline that ranks well is not an evaluator. These proxies read commanded
actions, not consequences. They exist to price the evaluator, and the claim
boundary says so.

## 6. Graded task-progress scores now have a producer

Blueprint already carried the whole consumer side of graded progress scoring —
rubric validation, five aggregation strategies, the aggregation ablation, the
blinded judge-calibration campaign. Nothing produced a score for them: every
live judge emitted a binary label.

This matters because the one ablation RoboWorld reports against its correlation
metric is the rubric itself (binary ρ = 0.922 versus graded ρ = 0.970), making
the rubric the cheapest demonstrated lever on rank fidelity. A binary label also
cannot separate "the policy failed" from "the world model fell apart".

`blueprint_pipeline.roboworld_progress_judge` adds:

- a **frame-sampling contract** that fails closed below 2.0 samples/second or 24
  frames, and refuses a segment carrying fewer than 4 samples. Six frames across
  a 25-second rollout is 0.24 fps — enough to guess a terminal state, useless for
  localising where progress stopped, and far too coarse for
  `progress_then_regression_aware` or `stable_maintenance`; and
- a conversion path that turns per-frame judge output into
  `roboworld_progress_score.v1` rows, each validated against the frozen rubric
  and its criterion-scoped view authority before emission.

The binary judges' frame budgets were raised from 5–6 to 16 for the same reason.

## 7. The public anchor: breaking the chicken-and-egg

Rank fidelity is gated on accepted real-world anchors, and `robot_eval_calibration`
rejects any real-world outcome that does not join to a prediction Blueprint
produced in the same job. Both rules are right for customer work. Together they
made the platform unprovable: demonstrating the evaluator required a customer
willing to buy an evaluator whose ranking had never been demonstrated.

Public real-world leaderboards already publish success rates for open policy
checkpoints, measured on physical robots by independent parties.

`blueprint_pipeline.public_benchmark_anchor` adds the two missing producers:

- `build_anchor_snapshot` canonicalises an operator-supplied leaderboard export
  into a digest-pinned snapshot. Its `snapshot_sha256` is the value the RoboWorld
  admission checklist already required as `roboarena_snapshot_sha256`, for which
  no producer existed.
- `build_external_reference_results` emits `external_reference_results.v1` — the
  schema `build_external_rank_fidelity_report` consumes, which previously had a
  constant and **no producer anywhere in the repository**.

Guardrails:

- Every policy row must carry a resolved `checkpoint_sha256`. A leaderboard names
  policies; it does not identify weights, and the fidelity report performs an
  exact-match join. For open policies this is satisfiable, so it is required.
- The module never asserts independence on the operator's behalf: an acceptance
  record with an accepting party, timestamp and source-artifact digest is
  required.
- `site_alignment` comes from the benchmark registry, not the caller, and a
  public distributed benchmark may never claim `same_site`.
- Duplicate checkpoints are rejected so two leaderboard rows cannot inflate the
  apparent cohort without adding a degree of freedom.

Results carry the distinct scope `harness_validation_public_anchor`. Passing
establishes that the scoring, aggregation, interval and ranking machinery
reproduces an independently measured real-world ordering **on the benchmark's own
embodiment, site distribution and task family**. `build_harness_validation_scope`
records what it does not establish — site-specific rank fidelity, any other
embodiment or action schema, physical task success or deployment readiness, and
world-model quality independent of the harness — and pins
`public_rank_fidelity_claim_eligible` to `False`.

## Commands

```bash
blueprint-run-control-ranker --input request.json --output report.json
blueprint-run-roboworld-progress-judge check-sampling --duration-seconds 25 --sampled-frames 60
blueprint-run-roboworld-progress-judge request --input plan.json --output request.json
blueprint-run-roboworld-progress-judge score --input judge-output.json --output scores.json
blueprint-build-public-benchmark-anchor snapshot --input export.json --output snapshot.json
blueprint-build-public-benchmark-anchor external-reference --input snapshot.json --output reference.json
```

No command launches a provider, trains a model, or allocates paid resources. The
progress judge reaches a model only through an operator-configured command, and
only when its gate env is set, a command is configured, and the request's
sampling contract marks it ready.
