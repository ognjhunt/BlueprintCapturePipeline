# RoboArena disjoint successor terminal report

Overall verdict: `inconclusive`.

Status note: this is an interim evidence snapshot, not the final campaign
closeout. A later independent first-principles audit found that the failed
DROID run lacked a known-good deployment positive control. The prospectively
frozen `phase_b_positive_control_bisection_amendment_v1.json` therefore
supersedes only the recommendation to stop future GPU diagnostics. It does not
alter the completed negative DROID result or increase its claim ceiling.

Seventeen frames were enough to test one narrow prerequisite: whether the
frozen native-Cosmos setup visibly followed a deliberately strong action chunk
better than valid no-motion and temporal-placebo controls. It failed that test
and Blueprint correctly abstained. Seventeen frames were not enough to assess a
complete task, long-horizon rollout stability, policy ranking, captured-site
transfer, or campaign economics. A short-screen pass would only have admitted
full episodes; it would not have supported the thesis by itself.

## Component verdicts

### cosmos_wam_qualification — inconclusive

The new native-Cosmos canary used one previously unused RoboArena session, the
actual task instruction, a prospectively selected high-motion real action
chunk, valid no-motion, shuffled, reversed, shifted, and real-policy-swapped
controls, and two seeds. Only `1/10` active same-seed comparisons separated
from no-motion, `2/10` rejected the strongest temporal placebo, and `0/5`
conditions passed both-seed robustness. The tier-1 reliability gate found five
hard-failure windows and a session-median timing correlation of `0.021936`
versus `0.15` required. Product abstention was therefore correct.

This was one independent session versus the prospective target of `17`, so it
does not adequately power a universal Cosmos falsification. It is valid
negative evidence for this frozen configuration only.

### frozen_benchmark_calibration — not supported for the frozen GPT-5-mini stack

The complete known-answer Phase-A reproduction covered `63` sessions, seven
policies, and `441` full OSCAR episodes. GPT-5 mini achieved Spearman `0.357143`,
Kendall tau-b `0.238095`, policy pairwise accuracy `0.619048` with clustered
interval `[0.428571, 0.857143]`, selective coverage `0.050182`, and selective
pairwise accuracy `0.600000`. It failed the registered gate set. Its claim
ceiling is a non-independent known-answer reproduction.

Gemini 3.6 Flash was promising on the already-unsealed matrix: Spearman `0.75`,
Kendall `0.619048`, policy pairwise accuracy `0.809524`, and direct within-session
accuracy `0.721925` with clustered interval `[0.676303, 0.762410]`. It never
abstained, so this diagnostic cannot admit confirmatory Phase B.

Cosmos3-Nano Reasoner loaded and returned exact-schema output on an H100, but
assigned zero progress while declaring stable success with confidence one. It
earned transport credit only, not evaluator or ranking credit.

### captured_site_transfer — inconclusive

Phase C was not admitted. The frozen stack was not run on Blueprint's own 3DGS
or site representation, and no independently attributable physical outcomes
were available for the same site, task, embodiment, and policies.

### economics_and_speed — inconclusive

Known conservative provider cost across calibration and successor work was
`$7.246017375`, excluding storage and transfer because no attributable invoice
line was available. This comprises `$6.175862375` in evaluator estimates and
`$1.070155` in GPU estimates, including `$0.24818` for the successful Reasoner
transport canary and `$0.088401` for the native-Cosmos causal canary.

The physical videos total `3.576` hours, but that is only a playback lower
bound. Setup, resets, failures, labor, robot cost, site preparation, and
parallelism were not independently anchored. Since no WAM arm completed a
useful ranking, no speed ratio, cost ratio, or break-even policy count is
claimed.

## Phase B actually achieved

The completed design was fallback level 3: a single-chunk, open-loop replay
causal diagnostic. It did not re-query a policy, create a terminal episode, or
rank policies. Sixteen selected sessions remain label-sealed.

Full policy-to-WAM-to-the-same-policy execution was not admitted because the
frozen Phase-A evaluator gates failed, native Cosmos failed its causal canary,
the public seven-policy checkpoints were not connected through a validated
observation/state/action adapter, and the selected sessions lacked attributable
camera calibration for OSCAR-style skeleton projection.

## WAM arms and historical baseline

Historical Experiment 2 remains `thesis_not_supported` for its frozen
OSCAR-derived stack. Its visible skeleton carried intended-motion signal, but
the skeleton-masked scene did not show useful action-conditioned dynamics.

The successor did not execute new skeleton-only, visible-skeleton,
skeleton-masked, OSCAR purpose-built WAM, or registered Cosmos-plus-skeleton
hybrid full episodes. Native Cosmos alone ran the one-session canary and failed.
No hybrid result is claimed, and Reasoner evaluator evidence remains separate
from WAM evidence.

## Identity, protocol, media, and provider state

The data were `RoboArena/DataDump_07-17-2026` revision
`7931db81f3f6a48a3245427f7213a4c461f92ccc`. OSCAR was pinned at
`4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`; Cosmos3-Nano at
`411f42a8fdfb8c5b2583cb8786e0938f49796eaa`; Phase A used
`gpt-5-mini-2025-08-07`; the challenger used `gemini-3.6-flash`.

Protocol v1 digest
`eab9e7868bcc7cbd774c940c781e8c3a8faac3270cbc942f1248966ba037f683`
remains immutable superseded history. Protocol v2 digest is
`6b41ea618ec290f1c080573e093d26cae03d14a0ecb06b3bb2a4bd016e469066`.

The non-cherry-picked single-view gallery and physical reference are indexed by
`phase_b_high_motion_native_cosmos_gallery_manifest_v1.json`. Raw three-camera
provider outputs remain retained separately; the composite was the native
Cosmos transport format, not an OSCAR view or a policy observation.

All task GPU instances were destroyed, authenticated inventory returned zero,
temporary task object prefixes and signed URLs were removed, no persistent
evaluator resource was created, and continuing hourly burn is zero. Provider
zero does not prove invoice settlement or scientific validity.

## Next finite experiment

Run the hash-pinned official NVIDIA AgiBotWorld four-chunk forward-dynamics
positive control. If it fails, submit zero DROID requests and teardown. If it
passes, run the already-frozen DROID control matrix in the same loaded process.
Only a subsequent powered test on genuinely new independent sessions can earn
DROID qualification or admit full-episode ranking; the positive control itself
is deployment diagnostics only.

In plain English: we proved that the harness can detect and reject a weak
short-horizon Cosmos prediction, and we found a judge that ranks the old public
benchmark promisingly. We did not complete the full policy-WAM-policy loop,
test Blueprint's own captured site, or establish a fair physical cost and speed
comparison. The evidence therefore does not support or conclusively falsify the
replaceable Blueprint thesis.
