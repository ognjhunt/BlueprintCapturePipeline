# RoboArena/DROID whole-stack calibration — 2026-07-28

Status: `phase_a_v4_schema_fix_ready_for_retry_after_zero_row_v3_failure`

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
and the preregistered uncertainty-aware risk/coverage rule. The complete
empirical and isotonic-smoothed curves must be published with
session-clustered bootstrap intervals. Adjacent empirical risk increases up to
0.02 are treated as numerical tolerance; a material statistically supported
increase as coverage falls fails the gate.

### Phase B — disjoint closed-loop confirmation

Only after Phase A passes, obtain a new independently labeled DROID/RoboArena
snapshot and run runnable frozen candidate-policy endpoints. For each WAM arm,
the loop is:

`policy -> action chunk -> one WAM -> new observation -> same policy`

The v1 value of 0.16 seconds was superseded before execution because 15 Hz makes
it a fractional 2.4 actions. The governing v2 prefix is 16 integer action steps,
with the duration derived as 16/15 = 1.0666666666666667 seconds. The selection
uses the exact pinned native Cosmos/DROID contract because live pilot endpoints
are not yet admitted: the reference path consumes one complete 16-action chunk
and emits the matching 16-frame future. A future label-free 4/8/16 canary is
still required before any different prefix could be registered. The loop
continues until task completion, safety or collapse abstention, or the frozen
maximum horizon. The scored artifact is the complete terminal episode, not one
short chunk.

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
`blueprint_pipeline.policy_ranking_roboarena_calibration`. Protocol v1 and its
digest
`eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683`
remain immutable, superseded pre-execution history. The amended v2 digest is
`6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066`.

The module rejects literal-zero rot6d controls, synthetic policy-swapped traces,
serial OSCAR/Cosmos chains, a short-chunk-only positive control, fractional or
mismatched closed-loop prefixes, and captured-site execution before the
disjoint benchmark gate. Static frames, first-frame collapse, repeated loops,
visual discontinuity, skeleton divergence, scene corruption, out-of-view
trajectories, increasing horizon uncertainty, and action-following degradation
are retained and count against reliability even when they trigger abstention.

## Current evidence boundary

This namespace currently proves only a corrected local protocol and label-sealed
inventory. The Phase A preflight re-fetched and hashed a ready 63-session ×
7-policy matrix containing 441 full OSCAR episodes without loading outcome
fields into the evaluator inventory. The all-video audit passed for 441/441
1280×480 sources and materialized 14,112 hashed 640×480 generated-only frames.
A deterministic 42-frame review sheet spanning all seven policies passed visual
leakage review. The label-free collapse audit retained all rows and flagged 39
episodes: 36 static/frozen, one repeated loop, and two sudden discontinuities.
An API credential supplied through ordinary chat was rejected, revoked, and
replaced through the task-scoped secure path; the replacement and its validated
user rotation attestation remain outside the repository. The 441-request GPT-5
mini inventory, prompt, strict schema, provider idempotency key, idempotent
result paths, and USD 22.05 conservative pre-call bound are frozen. The unpaid
v2 transport inventory is retained as superseded history; v3 changed only the
transport binding and derived digests, not the prompt, schema, sampling, or
scientific thresholds. The first v3 canary and a bounded text-only diagnostic
were rejected before inference because strict structured outputs did not accept
`uniqueItems`. They produced zero evaluator rows and accessed no outcomes. The
v4 amendment removes only that provider-invalid keyword and enforces identical
uniqueness locally; the prompt, semantic response contract, sampling, analysis,
and scientific thresholds are unchanged. The label-free prediction-freeze,
label-unseal, policy aggregation, exact-permutation uncertainty,
session-clustered bootstrap, calibration, and risk/coverage rules are bound by
analysis digest
`2b965b64c6894372cbcfa5091baacaa63f3c2300c21822770931b64ff3bd10eb`.
The v3 canary did call the provider and upload one generated-only episode; the
diagnostic uploaded fixed synthetic text only. No provider request completed
inference, and no evaluator result, new WAM generation, policy ranking, captured-site
transfer, or physical evaluation had occurred at that Phase-A checkpoint.
Invoice-attributable API cost was not yet available there, so the ledger
conservatively reserved USD 0.10 against the USD 25 API cap. A later paid
Reasoner v4 attempt used an estimated USD 0.1845 of GPU compute before failing
prior to model load; that attempt remains in campaign cost and reliability
accounting. Authenticated compute-provider inventory was zero at the Phase 0
audit and again after v4 teardown; provider zero is current resource-state
evidence, not a claim of zero historical spend.

## Post-unseal evaluator comparison continuation

The four-arm evaluator comparison remains diagnostic only: this public snapshot
and its physical answer key were already unsealed. Full GPT-5, GPT-5.4 mini,
Gemini 3.6 Flash, and Cosmos3-Nano Reasoner therefore cannot admit Phase B or
receive independent-confirmation credit here, even if they recover the public
leaderboard.

Cosmos Reasoner GPU execution is prospectively bound by
`cosmos_reasoner_gpu_execution_amendment_v1.json`, digest
`a6d7fc52a0bfb5a077b3bedc8a70ee6ebb1a8c75f0cd5dc313aec4b2ead94b78`.

The first paid Reasoner attempt is preserved as `cosmos3_reasoner_pilot_v4` in
the external evidence store. It failed before model loading or any pair result
because the runtime forced an unsupported architecture override. The
prospective, runtime-only supersession is
`cosmos_reasoner_runtime_compatibility_amendment_v2.json`. It retains the same
videos, pair selection, prompt, schema, decoding, budgets, and diagnostic-only
claim ceiling, while requiring the frozen model revision's native
`Cosmos3ForConditionalGeneration` architecture and rejecting future bundles
that does not exactly match the frozen runtime runner and entrypoint. Its
deterministic digest is
`ae10006cb17bcc58080a60b495755a53dd35fe64c2480a1559f2424546a550a1`.
It uses an evaluator-only provider bundle, one H100, a seven-pair canary, and a
two-hour/USD 5 pilot cutoff. A full matrix requires 7/7 valid pilot rows, a
frozen throughput projection that fits the USD 15 Reasoner-arm cap, and a new
single-use authorization. The existing Vast credential is accepted only under
the user's explicit task-scoped exception: live validity and mode 0600 are
proven, missing rotation metadata remains disclosed, and no provider-side
rotation event is claimed.

The exact-main v5 retry then loaded the pinned model and returned HTTP 200 for
all seven pairs, but produced zero valid scientific rows. The runner checked an
eleven-field schema after generation without transmitting that schema to vLLM;
all seven differently shaped responses were therefore rejected. This is a
transport-contract failure, not evaluator evidence, and the raw rejected
preferences are not reinterpreted. The prospective v3 supersession is
`cosmos_reasoner_structured_output_amendment_v3.json`. It binds vLLM's
documented `json_schema` response format, the provider-visible prompt schema,
the input manifest, and the post-generation validator to the same frozen schema
digest, failing before any provider request on drift. It applies to every future
Reasoner evaluator run and does not alter pair selection, field meanings,
scoring, thresholds, or the diagnostic-only claim ceiling.
Its deterministic amendment digest is
`e07bcde4f12ee38edc42d0e41546cc63bc33d27f926605cdbba9f9e89161f163`.
