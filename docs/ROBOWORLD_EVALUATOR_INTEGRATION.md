# RoboWorld-inspired evaluator integration

Blueprint implements the evaluator and measurement lessons from RoboWorld
behind model-neutral contracts. It does **not** implement Step Forcing, ship a
RoboWorld model adapter, inherit the paper's metrics, or claim that generated
video proves physical robot behavior.

The implementation surfaces are:

- `blueprint_pipeline.roboworld_evaluator`: progress rubric, criterion-scoped
  view authority, segment aggregation, segment ablation, judge calibration, and
  future-backend admission;
- `blueprint_pipeline.benchmark_uncertainty`: rollout convergence,
  policy-rank stability versus coverage, hierarchical uncertainty, and
  leave-one-out sensitivity;
- `roboworld_progress_v1` in `evaluator_evidence_profiles.py`: the evidence
  requirements for using the profile in decision-grade evaluator rows; and
- `_normalize_wam_success_labels()` in `oscar_cosmos_wam_evaluator.py`: the
  current WAM label seam, which preserves validated progress fields without
  turning them into real-world success.

The exact frozen default is tracked at
`docs/roboworld/roboworld_progress_evaluator_profile.v1.json`; tests require it
to equal the profile emitted by the implementation, including its canonical
digest.

## Progress rubric

`roboworld_progress_evaluator_profile.v1` freezes scores 0 through 5 while
keeping policy progress separate from world-model failure stage:

| Score | Progress interpretation | Typical model-error interpretation |
| --- | --- | --- |
| 0 | no task-directed progress | no creditable progress |
| 1 | approach | model failure during approach |
| 2 | approach without interaction | no required model failure |
| 3 | target contact | model failure at contact |
| 4 | near completion/substantial progress | possible failure during interaction |
| 5 | visibly completed and stably maintained | completion evidence must remain valid |

Every score carries:

- `task_progress_score`;
- `policy_progress_stage`;
- `world_model_failure_stage` and `world_model_failure_detected`;
- criterion and per-view evidence references;
- judge confidence and explicit abstention;
- prompt, judge-model, calibration-set, profile, and score digests.

The score remains generated-media review evidence. A score of 5 does not prove
physical task success, contact truth, safety, deployment readiness, or external
rank fidelity.

## Per-view authority

The default profile grants the two fixed external views authority for task
progress and completion. The wrist view can identify world-model failures but
cannot establish completion.

This is a default evaluator policy, not a universal belief that wrist views are
unreliable. A task-specific override is accepted only when it identifies one
criterion and view, supplies its allowed roles, cites an accepted calibration
set digest, records an explicit reason, and is independently accepted. An
unregistered role or an override without calibration fails closed.

## Segment aggregation

All strategies are computed side by side:

- `terminal`: last segment score;
- `mean`: arithmetic mean;
- `minimum`: worst segment score;
- `maximum_experimental`: best segment score;
- `progress_then_regression_aware`: terminal score minus accumulated drops
  between consecutive segment scores, clamped to 0;
- `stable_maintenance`: score 5 only when the configured number of final
  adjacent sampled frames all remain at score 5; otherwise terminal progress is
  retained but capped at 4.

`maximum_experimental` cannot be the profile default. The frozen pre-ablation
default is the conservative, directly interpretable `terminal` score;
`segment_aggregation_ablation.v1` compares
every strategy against independently accepted policy references using Pearson,
Spearman, Kendall tau-b, pairwise ordering accuracy, and MMRV. The report never
promotes a strategy automatically.

## Judge calibration campaign

`judge_calibration_campaign_request.v1` requires frozen samples, at least two
blinded and randomized human reviewers per sample, a GPT-family judge, an
independent Gemini-family judge, digest-bound prompts/models/calibration sets,
and at least three policies for rank analysis.

The resulting report includes:

- a six-by-six score confusion matrix for every judge;
- exact and within-one agreement;
- signed bias, MAE, abstention, false-success, Brier, and confidence-calibration
  results;
- Pearson, Spearman, Kendall, pairwise ordering, and MMRV against human policy
  aggregates; and
- bias breakouts by task, view condition, contact stage, and artifact type.

Human labels are the frozen campaign reference, not a substitute for matched
real-robot outcome anchors.

## Convergence and hierarchical uncertainty

`benchmark_uncertainty_report.v1` consumes digest-bound per-attempt predicted
and independently accepted reference scores. Its hierarchy resamples:

1. policies;
2. sites within sampled policy instances;
3. task families and tasks;
4. matched initial conditions; and
5. trials within the matched condition.

It reports 95% intervals for Pearson, Spearman, Kendall tau-b, pairwise ordering
accuracy, and MMRV; trial-count convergence; policy-rank stability versus
coverage; top-policy selection frequency; leave-one-policy-out sensitivity; and
leave-one-task-family-out sensitivity. Ten thousand bootstrap replicates are
required for the statistical layer to be eligible for the separate external
claim gate. The report itself never enables a public claim.

## Commands

```bash
blueprint-run-roboworld-evaluator-study profile --output profile.json
blueprint-run-roboworld-evaluator-study score --input score.json --output validated-score.json
blueprint-run-roboworld-evaluator-study aggregate --input segments.json --output aggregation.json
blueprint-run-roboworld-evaluator-study ablate-segments --input ablation.json --output ablation-report.json
blueprint-run-roboworld-evaluator-study calibrate-judges --input campaign.json --output campaign-report.json
blueprint-run-roboworld-evaluator-study admission --input admission-evidence.json --output checklist.json
blueprint-build-benchmark-uncertainty-report --input uncertainty.json --output uncertainty-report.json
```

No command launches a provider, trains a model, contacts a judge API, or
allocates paid resources. Judge outputs and human references are ingested as
evidence produced by separately authorized systems.

## Frozen upstream admission boundary

The admission checklist has four gates:

1. licensed, revision-pinned code, weights, container, preprocessing, data
   filters, action normalization, training schedule, and evaluation scripts;
2. reproduction of the BAIR Step Forcing diagnostic;
3. reproduction of the eight-policy, at-least-4,186-rollout RoboArena result;
4. a Blueprint-frozen identical matrix comparing the candidate with the current
   configured WAM, physics simulation, and external real anchors.

Until licensed code and weights exist, status is `awaiting_upstream_release`.
Paper-only Step Forcing reimplementation and model-backend integration remain
unauthorized by this contract. Even a fully admitted backend remains a
replaceable evaluation engine beneath Blueprint's capture, package, rights,
privacy, provenance, and buyer contracts.

## Schemas

- `docs/schemas/roboworld_progress_evaluator_profile.schema.json`
- `docs/schemas/roboworld_progress_score.schema.json`
- `docs/schemas/segment_aggregation_ablation.schema.json`
- `docs/schemas/judge_calibration_campaign_request.schema.json`
- `docs/schemas/judge_calibration_campaign_report.schema.json`
- `docs/schemas/benchmark_uncertainty_request.schema.json`
- `docs/schemas/benchmark_uncertainty_report.schema.json`
- `docs/schemas/roboworld_admission_evidence.schema.json`
- `docs/schemas/roboworld_admission_reproduction_checklist.schema.json`
