# RoboArena/DROID whole-stack calibration — 2026-07-28

Status: `preregistered_local_contract_only`

This is a new experiment. It does not reopen or modify the sealed OSCAR
Experiment 2 or Cosmos3 follow-up verdicts.

## Why the preceding Cosmos screen is not the requested full-stack test

The Cosmos3 follow-up used one public DROID observation, one 16-step recorded
action chunk, three temporal permutations/controls, a synthetic constant trace
labeled `policy_swapped`, and two seeds. Each response contained 17 frames at
15 FPS. It was a short WAM-conditioning screen, not a complete robot episode.

No candidate policy checkpoint or policy endpoint was queried. OSCAR was not
run. GPT-5 mini did not evaluate the clips. Blueprint did not aggregate episode
scores, rank policies, calibrate risk, or emit a policy-level abstention result.
The full scientific matrix did exercise the Cosmos request/runtime harness, but
it did not exercise the requested complete policy-ranking service.

The historical no-motion control was also malformed for its declared action
space: the six rot6d values were literal zeros instead of the identity rotation
`[1, 0, 0, 0, 1, 0]`, and the gripper value was assumed to be zero rather than
bound to an explicit hold state. The `policy_swapped` trace was synthetic rather
than an action trace from another real candidate policy. The original frozen
result remains `inconclusive`; these defects prevent promoting its descriptive
causal screen into a stronger conclusion about Cosmos3-Nano.

## New experiment sequence

### Phase A — public known-answer reproduction

Run Blueprint's independent evaluator, aggregation, ranking, uncertainty, and
abstention layers on the published full OSCAR/RoboArena episodes. Keep policy
identity and outcomes hidden from the evaluator and freeze predictions before
joining the published real-robot outcomes.

This phase is intentionally a reproduction because Blueprint has already used
the current 63 complete public sessions during method development. It can show
that the service reproduces the published benchmark from full episodes. It
cannot be called independent confirmation or captured-site transfer.

All registered endpoint gates must pass: Spearman rho at least 0.70, Kendall
tau-b at least 0.50, pairwise accuracy at least 0.70 with clustered 95% lower
bound at least 0.50, the real top policy within Blueprint's predicted top two,
selective coverage at least 0.50, selective pairwise accuracy at least 0.75,
and non-increasing risk as the service abstains more aggressively.

### Phase B — disjoint closed-loop confirmation

Only after Phase A passes, obtain a new independently labeled DROID/RoboArena
snapshot and run runnable frozen candidate-policy endpoints. For each WAM arm,
the loop is:

`policy -> action chunk -> one WAM -> new observation -> same policy`

Only 0.16 seconds of the predicted action horizon is advanced before the policy
is queried again. The loop continues until task completion, safety abstention,
or the frozen maximum horizon. The scored artifact is the complete terminal
episode, not one short chunk.

OSCAR and Cosmos3 receive the same frozen inputs as parallel attributable arms.
They never feed outputs into each other. OSCAR is the public purpose-built
baseline. Cosmos3 is diagnostic until its corrected v2 controls and independent
causal qualification pass.

### Phase C — captured-site transfer

Captured-site/3DGS evaluation remains blocked until every Phase B rank and
abstention gate passes. A site-specific accuracy claim additionally requires
independently published physical outcomes for that site and task. A plausible
generated episode or a 3DGS render alone is not accuracy evidence.

## Frozen implementation

The executable protocol is
`blueprint_pipeline.policy_ranking_roboarena_calibration`. Its deterministic
protocol digest is
`eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683`.

The module rejects literal-zero rot6d controls, synthetic policy-swapped traces,
serial OSCAR/Cosmos chains, a short-chunk-only positive control, a missing
0.16-second closed-loop prefix, and captured-site execution before the disjoint
benchmark gate.

## Current evidence boundary

This namespace currently proves only that the corrected local protocol compiles
and its focused tests pass. The Phase A preflight also materialized and hashed a
ready 63-session × 7-policy matrix containing 441 full OSCAR episodes without
loading outcome fields into the evaluator inventory. The secure local API-key
destination was not approved, so no key was created and no provider call or
upload occurred. No new WAM generation, evaluator result, policy ranking,
captured-site transfer, physical evaluation, or provider allocation has
occurred in this experiment yet.
