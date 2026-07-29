# RoboArena disjoint successor terminal report v2

Overall verdict: `inconclusive`.

The finite positive-control bisection is complete. The pinned Cosmos3-Nano
deployment produced dynamic forward predictions on NVIDIA's published
AgiBotWorld pathway: three of four chunks passed the frozen motion gate. The
same loaded runtime then failed Blueprint's frozen DROID causal screen. This
separates a generally broken deployment from a DROID conditioning failure: the
model and GPU path worked, but the tested robot-action interface was not
reliable enough for Blueprint to trust.

## Component verdicts

### cosmos_wam_qualification — inconclusive

Native Cosmos passed the official deployment positive control. On the DROID
matrix, only `1/10` active same-seed comparisons separated from no motion,
`2/10` rejected their strongest temporal placebo, and `0/5` conditions were
robust across both seeds. The integrated tier-1 reliability gate found five
`static_under_command` failures and a session-median timing correlation of
`0.022880`, below the frozen `0.15` minimum. Blueprint correctly abstained.

This is valid negative evidence for the frozen DROID configuration, but only
one independent session was tested versus the prospective target of 17. It is
therefore underpowered for general Cosmos WAM falsification.

### frozen_benchmark_calibration — frozen GPT-5-mini stack not supported

Phase A was a non-independent known-answer reproduction over 63 sessions,
seven policies, and 441 full OSCAR episodes. GPT-5 mini achieved Spearman
`0.357143`, Kendall tau-b `0.238095`, policy pairwise accuracy `0.619048` with
clustered interval `[0.428571, 0.857143]`, selective coverage `0.050182`, and
selective pairwise accuracy `0.600000`. It failed the registered gate set.

Gemini 3.6 Flash was promising only as an already-unsealed diagnostic:
Spearman `0.75`, Kendall tau-b `0.619048`, policy pairwise accuracy `0.809524`,
and direct within-session accuracy `0.721925` with clustered interval
`[0.676303, 0.762410]`. It never abstained, so it cannot admit confirmatory
Phase B on this exposed matrix. The Cosmos3-Nano Reasoner canary remains a
transport-only result because its response was semantically inconsistent.

### captured_site_transfer — inconclusive

Phase C was not admitted. The stack was not run on Blueprint's 3DGS/site
representation, and no independently attributable outcomes existed for the
same site, task, embodiment, and candidate policies.

### economics_and_speed — inconclusive

Known conservative provider spend is `$7.856395375`: `$6.175862375` in
evaluator estimates, `$1.070155` in earlier GPU estimates, `$0.463386` for the
stalled positive-control download attempt, and `$0.146992` for the successful
bisection. Storage and transfer lack an attributable invoice line. The
successful retry loaded the model in `190.063` seconds, ran in `258.269`
seconds, and occupied the provider for approximately `492.671` seconds.

The physical videos total `3.576` hours, but that is only a playback lower
bound. It excludes setup, resets, failures, labor, robot cost, site preparation,
and parallelism. Because no WAM arm completed a useful ranking, no speed ratio,
cost ratio, or break-even policy count is claimed.

## What Phase B actually achieved

The achieved design was fallback level 3: a one-session open-loop replay of
frozen real actions and controls. It did not re-query a runnable policy, create
a complete terminal episode, call an evaluator, or rank policies. Sixteen
selected sessions remain label-sealed. The causal and reliability failures
prevented admission to the expensive full-episode matrix.

The desired positive result would have been visible and directionally correct
response to the recorded action, clear separation from valid no-motion,
shuffled, reversed, shifted, and real-policy-swapped controls, robustness over
both seeds, and a reliability gate that admitted the outputs. That would have
qualified this WAM configuration to attempt full episodes. It still would not
have proven policy-ranking accuracy or the overall thesis without new physical
outcome anchors.

## WAM arms and historical baseline

Historical Experiment 2 remains `thesis_not_supported` for its frozen
OSCAR-derived stack. Its visible skeleton carried intended-motion signal, but
the skeleton-masked scene did not establish useful action-conditioned world
dynamics. No new skeleton-only, visible-skeleton, skeleton-masked, OSCAR WAM,
or registered Cosmos-plus-skeleton hybrid episode ran because the selected
snapshot lacked attributable camera calibration. Native Cosmos alone ran the
new bisection; no hybrid result is claimed.

## Identity, protocols, media, and provider state

Execution used source `8364c2f2b811f9db1142b9a5c86839a6b6b01a80`,
Cosmos3-Nano revision `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`,
vLLM-Omni revision `9c1b7504b178afcf541867c1a2d30db48c69cda8`, and
runtime image digest
`sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.
The DROID data source remains `RoboArena/DataDump_07-17-2026` revision
`7931db81f3f6a48a3245427f7213a4c461f92ccc`; OSCAR is pinned at
`4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`.

Protocol v1 digest
`eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683`
remains immutable superseded history. Governing v2 digest is
`6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066`.
The download repair amendment digest is
`ae034d9ebd976c4f4d5540d340182d69568555bf772b18fd4b6bc6544bfbc7fc`.

The non-cherry-picked 32-video gallery is indexed as
`external-evidence-store://policy-ranking-roboarena-disjoint-reasoner-successor-evidence-20260728/phase_b_positive_control_v2/review_gallery_v1`.
It contains every DROID condition and seed, all four official positive-control
chunks, the original three-camera composites, and top-camera review crops. The
crops were not used for scientific metrics.

All task GPU instances were destroyed. Authenticated inventory returned zero
task and total instances, continuing hourly burn is zero, temporary object
prefixes are absent, six signed-URL files were removed, and no persistent
evaluator resource was created. Provider zero does not prove invoice settlement
or scientific validity.

## Cheapest valid next experiment

Audit the exact upstream Cosmos3 DROID normalization and camera/action
preprocessing. If it remains ambiguous, run the already-preregistered,
label-free action-scale dose-response diagnostic. Freeze any resulting transform
before seeing outcomes, then test it on new untouched independent DROID sessions
with adequate power. Only a causal pass should admit complete policy ranking.

In plain English: Cosmos itself was capable of generating motion, but it did not
reliably obey our DROID robot commands. Blueprint noticed that and refused to
trust the videos. We did not complete the policy-to-WAM-to-policy loop, rank new
policies, test a captured Blueprint site, or prove a physical cost advantage.
